'''
This script is used to evaluate the consistency of the agents' activities with the specific event.
'''
import os
import json
from datetime import datetime, timedelta
import argparse
from scripts.llm_real import OpenAILLM
from utils import *
from collections import Counter

parser = argparse.ArgumentParser(description='run humanoid agents simulation')
parser.add_argument("-c", "--config_filename", default="../simulation_args/fitness.json")


args = parser.parse_args()
args = load_json_file(args.config_filename)

input_folder_name = args["input_folder_name"]
output_folder_name = args["output_folder_name"]
agent_filenames = args["agent_filenames"]
llm_model_name = args["llm_model_name"]
embedding_model_name = args["embedding_model_name"]

evalLLM = OpenAILLM(llm_model_name=llm_model_name, embedding_model_name=embedding_model_name)

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
                    if histories[agent] == [] or histories[agent][-1][1] != data[agent]["activity"]:
                        if histories[agent] != []:
                            bag_info = histories[agent][-1][-1]
                            # histories[agent][-1][-1] = f"to {current_time.strftime('%I:%M:%S %p')}"
                            # histories[agent][-1].append(bag_info)
                        bag = [item[0] for item in data[agent]["bag"]]
                        bag_info = "with bag:" + ",".join(bag)
                        histories[agent].append([f"{current_time.strftime('%I:%M:%S %p')}:", data[agent]["activity"]])
        
        current_time += time_increment



# Example usage
folder_path = '../generations/fitness_2'
check_history(folder_path)

output = []
for item in histories["Mei Lin"]:
    if "is on the way to" in item[1]:
        continue
    print(f"{item[0]} {item[1]}")

# for agent in agent_filenames:
#     print("\n"*5)
#     print(f"Agent {agent} has the following activities:")
#     print('\n'.join([" ".join(item) for item in histories[agent]]))
#     history = '\n'.join([" ".join(item) for item in histories[agent]])

#     eval_prompt = f"""Given the following activities, rate whether the action history of {agent} is consistent to the specific event: Isabella Rodriguez is planning to invite her friends Klaus Mueller and Maria Lopez to dinner at Hobbs Cafe from 5 pm to 7 pm. Isabella will be responsible for bringing milk and bread.
# Please rate their confirmity based on their action on a scale of 1 to 10, with 1 being definitely not consistent and 10 being definitely consistent.
# Please only focus on the position of the agent, they need to stay in Hobbs Cafe during 5 pm to 7 pm.
# Also 
# Please rate the correlation between the agent's action and the items in the bag.
# Here is the action history of {agent}:
# {history}
# Answer: """
#     answer = evalLLM.get_llm_response(eval_prompt)
#     print(answer)
    
    
# '''

# 硬性：出席，带东西，
# 弹性：action必须相关，日常，日常的package是否合理
# test package reasonableness
# daily plan requirement
# '''