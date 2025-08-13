import os
import json
from datetime import datetime, timedelta
import argparse
from utils import *
from collections import Counter
from llm import OpenAILLM

parser = argparse.ArgumentParser(description='run humanoid agents simulation')
parser.add_argument("-c", "--config_filename", default="../simulation_args/philosophy_lecture/1.json")

args = parser.parse_args()
args = load_json_file(args.config_filename)

input_folder_name = args["input_folder_name"]
output_folder_name = args["output_folder_name"]
agent_filenames = args["agent_filenames"]

LLM = OpenAILLM(llm_model_name=args["llm_model_name"], embedding_model_name=args["embedding_model_name"])

histories = {}
noteworthy = ""
def post_process(folder):
    if os.path.exists(os.path.join(folder, "records_for_sim.json")):
        record_filename = os.path.join(folder, "records_for_sim.json")
    else:
        record_filename = os.path.join(folder, "records.json")
    print(record_filename)
    total_data = json.load(open(record_filename, 'r'))
    end_time = datetime.strptime("08:00:00 pm", "%I:%M:%S %p")
    for i, (time, data) in enumerate(total_data.items()):
        if datetime.fromisoformat(time).time() > end_time.time():
            break
        # print(data.keys())
        print(time)
        # return
        statements = []
        for agent in agent_filenames:
            if agent not in data:
                data[agent] = histories[agent]
            histories[agent] = data[agent]
            statement = data[agent]["activity"]
            statements.append(statement)
            name = statement.split(" is ")[0]
            initial_letters = ' '.join([f"{item[0]}." for item in name.split()])
            statement = statement.replace(name, initial_letters)
            if "is on the way" in statement:
                statement = initial_letters + " is on the way."
            elif "is talking" in statement:
                statement = initial_letters + " is in a conversation."
            else:
                if len(statement.split(" for ")) > 2:
                    statement = "for".join(statement.split(" for ")[0:1])
            
                statement = statement.split(" from ")[0]
                statement = statement.split(" in ")[0]
                statement = statement.split(" at ")[0]
                if len(statement.split(' to ')) > 2:
                    statement = ' to '.join(statement.split(' to ')[0:1])
            
            data[agent]["short_activity"] = statement
            bag = []
            for item in data[agent]["bag"]:
                if type(item) is list:
                    bag.append(item[0])
                else:
                    bag.append(item)
            data[agent]["bag"] = bag
        #     else:
        #         data[agent] = last_data[agent]
        # last_data = data
        
        # per 10min 
        
        if 'noteworthy' in data:
            noteworthy = str(data['noteworthy'])
        if i % 10 == 0 and 'noteworthy' not in data:
        
            noteworthy_prompt = f'''At {time}, all agents are doing the following activities:\n
{', '.join(statements)}.\n
Please identify the at most three noteworthy events happening now, such as groups of agents gathering in the same location or multiple agents engaging in similar activities.
Your response should be concise and focus on the most significant events that involve multiple agents.
For each event, provide a brief description of the activity and the agents involved.
Your response should be a array with the following format:
[
    {{"event": "<event 1>", "people": ["<agent>", "<agent>"]}},
    {{"event": "<event 2>", "people": ["<agent>", "<agent>"]}},
    {{"event": "<event 3>", "people": ["<agent>"]}}
]
The description of the event should be short and concise, focusing on the activity.
Focus on the agents' activities and interactions, not the environment.
Output only the array in plain text, do not include any other text or formatting.
    '''
            # print(noteworthy_prompt)
            noteworthy = LLM.get_llm_response(noteworthy_prompt, max_tokens=500)
        # noteworthy = noteworthy.strip().split('\n')
            print(noteworthy)
        # print(data["noteworthy"])
        data["noteworthy"] = eval(noteworthy)
    
    
        if i % 2000 == 0:
            json.dump(total_data, open(os.path.join(folder, "records_for_sim.json"), 'w'), indent=4)
    json.dump(total_data, open(os.path.join(folder, "records_for_sim.json"), 'w'), indent=4)

# Example usage
folder_path = '../generations/philosophy_lecture'
post_process(folder_path)
