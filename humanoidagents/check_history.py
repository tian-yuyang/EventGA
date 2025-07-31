import os
import json
from datetime import datetime, timedelta
import argparse
from utils import *
from collections import Counter

parser = argparse.ArgumentParser(description='run humanoid agents simulation')
parser.add_argument("-c", "--config_filename", default="../simulation_args/fitness.json")


args = parser.parse_args()
args = load_json_file(args.config_filename)

input_folder_name = args["input_folder_name"]
output_folder_name = args["output_folder_name"]
agent_filenames = args["agent_filenames"]

histories = {}
def check_history(folder):
    start_time = datetime.strptime("07:00:00 am", "%I:%M:%S %p")
    end_time = datetime.strptime("07:00:00 pm", "%I:%M:%S %p")
    time_increment = timedelta(seconds=5)
    current_time = start_time
    
    while current_time <= end_time:
        filename = f"{current_time.strftime('%I_%M_%S %p')}.json"
        filepath = os.path.join(folder, filename)
        
        if not os.path.exists(filepath):
            break
        with open(filepath, 'r') as file:
            data = json.load(file)
            for agent in agent_filenames:
                if agent in data:
                    # histories[agent] = histories.get(agent, Counter())
                    # histories[agent][data[agent]["activity"]]+=1
                    histories[agent] = histories.get(agent, [])
                    if histories[agent] != []: 
                        if abs(histories[agent][0] - data[agent]["curr_tile"][0]) + abs(histories[agent][1] - data[agent]["curr_tile"][1]) > 1:
                            print("Wrong movement")
                        histories[agent] = data[agent]["curr_tile"]
        
        current_time += time_increment

# Example usage
folder_path = '../generations/fitness_2'
check_history(folder_path)

for agent in agent_filenames:
    print("\n"*5)
    print(f"Agent {agent} has the following activities:")
    print(histories[agent])

# for agent in agent_filenames:
#     print("\n"*5)
#     print(f"Agent {agent} has the following activities:")
#     print('\n'.join([" ".join(item) for item in histories[agent]]))

# for agent in agent_filenames:
#     on_the_way = 0
#     print(f"Agent {agent} has the following activities:")
#     for k, v in histories[agent].items():
#         print(f"{v} count: {k}")
#         if "is on the way" in k:
#             on_the_way += v
#     print(f"Agent {agent} was on the way {on_the_way} times.")