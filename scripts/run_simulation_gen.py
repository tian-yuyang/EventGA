import argparse
import json
import logging
import os
from datetime import datetime

from tqdm import tqdm

import heapq
from scripts.llm_real import openai_api_deploy
from generative_agent_gen import GenerativeAgent
from location import Location
from maze_new import Maze
from utils import *
import shutil, errno

def copyanything(src, dst):
  try:
    shutil.copytree(src, dst)
  except OSError as exc: # python >2.5
    if exc.errno in (errno.ENOTDIR, errno.EINVAL):
      shutil.copy(src, dst)
    else: raise
    
    
logging.basicConfig(format='---%(asctime)s %(levelname)s \n%(message)s ---', level=logging.INFO)
logging.getLogger("openai").setLevel(logging.WARNING)

parser = argparse.ArgumentParser(description='run humanoid agents simulation')
parser.add_argument("-c", "--config_filename", default="../simulation_args/imk.json")


args = parser.parse_args()
args = load_json_file(args.config_filename)
logging.info(args)

input_folder_name = args["input_folder_name"]
output_folder_name = args["output_folder_name"]
if not os.path.exists(output_folder_name):
    copyanything(input_folder_name, output_folder_name)
if not os.path.exists(f"{output_folder_name}/timesteps"):
    os.makedirs(f"{output_folder_name}/timesteps")
if not os.path.exists(f"{output_folder_name}/states"):
    os.makedirs(f"{output_folder_name}/states")
agent_filenames = args["agent_filenames"]
start_time = args["start_time"]
default_agent_config_filename = args["default_agent_config_filename"]
llm_provider = args["llm_provider"]
llm_model_name = args["llm_model_name"]
embedding_model_name = args["embedding_model_name"]
daily_events_filename = args["daily_events_filename"]
end_time = args["end_time"]


## location
maze = Maze()

agents = []

curr_time_to_daily_event = get_curr_time_to_daily_event(daily_events_filename)
print(curr_time_to_daily_event)

default_agent_kwargs = load_json_file(default_agent_config_filename)

for agent_filename in agent_filenames:
    agent_kwargs = {"path":f"{output_folder_name}/{agent_filename}"}
    #inplace dict update
    agent_kwargs["llm_provider"] = llm_provider
    agent_kwargs["llm_model_name"] = llm_model_name
    agent_kwargs["embedding_model_name"] = embedding_model_name
    agent_kwargs["action_end_time"] = datetime.fromisoformat(start_time)
    agent = GenerativeAgent(**agent_kwargs)
    heapq.heappush(agents,agent)
    
## daily_events

records = {}
time_slice_records = {}
if(os.path.exists(f"{output_folder_name}/records.json")):
    records = load_json_file(f"{output_folder_name}/records.json")
for k, v in records.items():
    time_slice_records[datetime.fromisoformat(k)] = v
simulation_time = datetime.fromisoformat(start_time)
end_time = datetime.fromisoformat(end_time)

agent_save_step = 5

# run simulation
while agents:
    agent = heapq.heappop(agents)
    curr_time = agent.action_end_time
    
    condition = ''
    if curr_time.strftime("%y-%m-%d") in curr_time_to_daily_event and agent.name in curr_time_to_daily_event[curr_time.strftime("%y-%m-%d")]:
      condition = curr_time_to_daily_event[curr_time.strftime("%y-%m-%d")][agent.name]
    if simulation_time > end_time:
        heapq.heappush(agents,agent)
        break
    if curr_time <= simulation_time:
        agent_save_step -= 1
        logging.info(curr_time)
        overall_status = agent.get_status_json(curr_time, maze, time_slice_records, agents, condition)
        
        if type(overall_status) == list:
            overall_status, talking_agent_status = overall_status
            overall_log = {
                "date": DatetimeNL.get_date_nl(curr_time),
                "time": DatetimeNL.get_time_nl(curr_time),
                "agents": talking_agent_status,
            }
            output_filename = f"{output_folder_name}/states/state_{talking_agent_status['name']}_{DatetimeNL.get_time_nl(curr_time).replace(':','_')}.json"
            print(output_filename)
            write_json_file(overall_log, output_filename)

        logging.info("Overall status:")
        logging.info(json.dumps(overall_status, indent=4))

            
        overall_log = {
            "date": DatetimeNL.get_date_nl(curr_time),
            "time": DatetimeNL.get_time_nl(curr_time),
            "agents": overall_status,
        }
        output_filename = f"{output_folder_name}/states/state_{agent.name}_{DatetimeNL.get_time_nl(curr_time).replace(':','_')}.json"
        print(output_filename)
        write_json_file(overall_log, output_filename)
    else:
        output_filename = f"{output_folder_name}/timesteps/{DatetimeNL.get_time_nl(simulation_time).replace(':','_')}.json"
        write_json_file(time_slice_records[simulation_time], output_filename)
        simulation_time = DatetimeNL.add_time_sec(simulation_time, step_size)
    if agent_save_step <= 0:
        agent.save(f"{output_folder_name}/{agent.name}")
        for agent_ in agents:
            agent_.save(f"{output_folder_name}/{agent_.name}")
        
        records = {}
        for time, record in time_slice_records.items():
            records[datetime.isoformat(time)] = record
        write_json_file(records, f"{output_folder_name}/records.json")
        agent_save_step = 5
        
    heapq.heappush(agents,agent)
print("Simulation ended")
for agent in agents:
    agent.save(f"{output_folder_name}/{agent.name}")
    
records = {}
for time, record in time_slice_records.items():
    records[datetime.isoformat(time)] = record
write_json_file(records, f"{output_folder_name}/records.json")





"""

evaluation: 出席率，把活动历史让模型判断一天的好坏，item是否合理

1 所有人知道所有事 80+
2 一个人知道所有事 分配任务 50+
3 地点或时间模糊 物品要么都一样要么都不一样 10+
"""