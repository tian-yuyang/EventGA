# '''
# This script is used to evaluate the consistency of the agents' activities with the specific event.
# '''
# import os
# import json
# from datetime import datetime, timedelta
# import argparse
# from llm import OpenAILLM
# from utils import *
# from collections import Counter

# parser = argparse.ArgumentParser(description='run humanoid agents simulation')
# parser.add_argument("-c", "--config_filename", default="../simulation_args/fitness.json")


# args = parser.parse_args()
# args = load_json_file(args.config_filename)

# input_folder_name = args["input_folder_name"]
# output_folder_name = args["output_folder_name"]
# agent_filenames = args["agent_filenames"]
# llm_model_name = args["llm_model_name"]
# embedding_model_name = args["embedding_model_name"]

# evalLLM = OpenAILLM(llm_model_name=llm_model_name, embedding_model_name=embedding_model_name)

# histories = {}
# def check_history(folder):
#     start_time = datetime.strptime("07:00:00 am", "%I:%M:%S %p")
#     end_time = datetime.strptime("07:00:00 pm", "%I:%M:%S %p")
#     time_increment = timedelta(seconds=5)
#     current_time = start_time

#     while current_time <= end_time:
#         filename = f"{current_time.strftime('%I_%M_%S %p')}.json"
#         filepath = os.path.join(folder, filename)
        
#         if not os.path.exists(filepath):
#             break
#         with open(filepath, 'r') as file:
#             data = json.load(file)
#             for agent in agent_filenames:
#                 if agent in data:
#                     # histories[agent] = histories.get(agent, Counter())
#                     # histories[agent][data[agent]["activity"]]+=1
#                     histories[agent] = histories.get(agent, [])
#                     if histories[agent] == [] or histories[agent][-1][0] != data[agent]["activity"]:
#                         if histories[agent] != []:
#                             bag_info = histories[agent][-1][-1]
#                             histories[agent][-1][-1] = f"to {current_time.strftime('%I:%M:%S %p')}"
#                             histories[agent][-1].append(bag_info)
#                         bag = [item[0] for item in data[agent]["bag"]]
#                         bag_info = "with bag:" + ",".join(bag)
#                         histories[agent].append([data[agent]["activity"],f"from {current_time.strftime('%I:%M:%S %p')} ", bag_info])
        
#         current_time += time_increment



# # Example usage
# folder_path = '../generations/fitness_2'
# check_history(folder_path)

# for agent in agent_filenames:
#     print("\n"*5)
#     print(f"Agent {agent} has the following activities:")
#     print('\n'.join([" ".join(item) for item in histories[agent]]))
#     history = '\n'.join([" ".join(item) for item in histories[agent]])

#     eval_prompt = f"""Given the following activities, rate whether the action history of {agent} is consistent to the specific event: Wolfgang Schulz plans to organize a fitness competition at Oak Hill College. The competition will start at 11 am and end at 2 pm. Wolfgang Schulz invites Francisco Lopez, Ayesha Khan and Klaus Mueller to be there, and Wolfgang Schulz is responsible for bringing water and bread.
# Please rate their confirmity based on their action on a scale of 1 to 10, with 1 being definitely not consistent and 10 being definitely consistent.
# Please focus on the position of the chosen agent, they need to stay in Oak Hill College during the competition.
# Also Please rate the correlation between the agent's action and the items in the bag.
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


import os
import json
from datetime import datetime, timedelta
import argparse
from scripts.llm_real import OpenAILLM
from generative_agent_gen import GenerativeAgent
from utils import *

# 解析命令行参数
parser = argparse.ArgumentParser(description='evaluation')
parser.add_argument("-c", "--config_filename", default="../simulation_args/fitness.json")
args = parser.parse_args()
args = load_json_file(args.config_filename)

# 获取配置参数
input_folder_name = args["input_folder_name"]
output_folder_name = args["output_folder_name"]
agent_filenames = args["agent_filenames"]
llm_model_name = args["llm_model_name"]
embedding_model_name = args["embedding_model_name"]

# 初始化语言模型
evalLLM = OpenAILLM(llm_model_name=llm_model_name, embedding_model_name=embedding_model_name)

# 定义代理角色
roles = {
    "Wolfgang Schulz": "organizer",
    "Francisco Lopez": "participant",
    "Ayesha Khan": "participant",
    "Klaus Mueller": "participant"
}


criteria = [
    "Role Fulfillment",
    "Location Adherence",
    "Bag Item Relevance",
    "Daily Requirement Consistency",
    "Interaction Quality"
]

def get_prompt(agent, criterion, history):
    role = roles.get(agent,"")
    if criterion == "Role Fulfillment":
        if role == "organizer":
            return f"Evaluate how well {agent} fulfilled his role as the organizer of the fitness competition at Oak Hill College. Consider whether he organized the event, invited the participants (Francisco Lopez, Ayesha Khan, Klaus Mueller), brought water and bread, and managed the event during 11 am to 2 pm. Rate on a scale of 1 to 10, with 1 being poor fulfillment and 10 being excellent fulfillment. Here is {agent}'s action history:\n{history}\nPlease provide your answer as a single number between 1 and 10."
        elif role == "participant":
            return f"Evaluate how well {agent} fulfilled their role as a participant in the fitness competition at Oak Hill College. Consider whether they attended the event from 11 am to 2 pm and participated in the competition. Rate on a scale of 1 to 10, with 1 being poor participation and 10 being excellent participation. Here is {agent}'s action history:\n{history}\nPlease provide your answer as a single number between 1 and 10."
        else:
            return None
    elif criterion == "Location Adherence":
        return f"Evaluate whether {agent} was at Oak Hill College during the competition hours from 11 am to 2 pm. Rate on a scale of 1 to 10, with 1 being not present at all and 10 being present throughout the entire period.\n Here is {agent}'s action history:\n{history}\nPlease provide your answer as a single number between 1 and 10."
    elif criterion == "Bag Item Relevance":
        return f"Evaluate whether the items in {agent}'s bag are relevant to their actions during the day. For example, if they are participating in a fitness competition, do they have sports gear? If they are organizing, do they have organizational materials? Rate on a scale of 1 to 10, with 1 being no relevance and 10 being highly relevant.\n Here is {agent}'s action history:\n{history}\nPlease provide your answer as a single number between 1 and 10."
    elif criterion == "Daily Requirement Consistency":
        
    elif criterion == "Daily Activity Relevance":
        return f"Evaluate whether {agent}'s activities throughout the day are relevant to their daily plan requirement. For the organizer, this includes preparation, organization, and management. For participants, this includes preparation and participation. Rate on a scale of 1 to 10, with 1 being completely irrelevant and 10 being highly relevant.\nHere is {agent}'s personal information: {agents[agent].get_str_iss()}\n Here is {agent}'s action history:\n{history}\nPlease provide your answer as a single number between 1 and 10."
    elif criterion == "Interaction Quality":
        return f"Evaluate the quality of {agent}'s interactions with other agents. If it is related to the specific event, the score should be high. Consider whether they communicated effectively, coordinated actions, and collaborated as needed for their role. Rate on a scale of 1 to 10, with 1 being poor interaction and 10 being excellent interaction. Here is {agent}'s action history:\n{history}\nPlease provide your answer as a single number between 1 and 10."
    else:
        return None

agents = {}
for agent_filename in agent_filenames:
    agent_kwargs = {"path":f"{output_folder_name}/{agent_filename}"}
    agent = GenerativeAgent(**agent_kwargs)
    agents[agent_filename] = agent

histories = {}
def check_history(folder):
    start_time = datetime.strptime("06:00:00 am", "%I:%M:%S %p")
    end_time = datetime.strptime("06:00:00 pm", "%I:%M:%S %p")
    time_increment = timedelta(seconds=5)
    current_time = start_time

    while current_time <= end_time:
        filename = f"{current_time.strftime('%I_%M_%S %p')}.json"
        filepath = os.path.join(folder, "timesteps", filename)
        
        if not os.path.exists(filepath):
            break
        with open(filepath, 'r') as file:
            data = json.load(file)
            for agent in agent_filenames:
                if agent in data:
                    histories[agent] = histories.get(agent, [])
                    if histories[agent] == [] or histories[agent][-1][0] != data[agent]["activity"]:
                        if histories[agent] != []:
                            bag_info = histories[agent][-1][-1]
                            histories[agent][-1][-1] = f"to {current_time.strftime('%I:%M:%S %p')}"
                            histories[agent][-1].append(bag_info)
                        bag = [item[0] for item in data[agent]["bag"]]
                        bag_info = "with bag:" + ",".join(bag)
                        histories[agent].append([data[agent]["activity"], f"from {current_time.strftime('%I:%M:%S %p')} ", bag_info])
        
        current_time += time_increment

# 示例使用
folder_path = '../generations/fitness_2'
check_history(folder_path)

# 评估每个代理
for agent in agent_filenames:
    # print("\n"*5)
    # print(f"history of {agent}: ")
    history_list = histories.get(agent, [])
    history_str = '\n'.join([" ".join(map(str, item)) for item in history_list])
    # print(history_str)

    print(f"\n{agent} evaluation results:")
    for criterion in criteria:
        prompt = get_prompt(agent, criterion, history_str)
        if prompt is None:
            continue
        answer = evalLLM.get_llm_response(prompt)
        print(f"{criterion}: {answer}")