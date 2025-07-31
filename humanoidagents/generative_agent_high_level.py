import json
import logging
from datetime import datetime
from functools import cache
import os
import random
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from operator import itemgetter
from llm import OpenAILLM, LocalLLM, MindsDBLLM
from utils import *
from spatial_memory import MemoryTree
from path_finder import *

class GenerativeAgent:

    def __init__(self, path: str, 
                 llm_provider: str, llm_model_name: str, embedding_model_name: str, action_end_time: datetime):
        self.name = ''
        self.description = ''
        self.daily_plan_req = ''
        self.currently = ''
        self.learned = ''
        self.lifestyle = ''
        self.age = ''
        self.traits = ''
        self.curr_tile = ''
        self.social_relationships = ''
        self.reflection_frequency = reflection_frequency
        self.talking_frequency = talking_frequency
        self.action_end_time = action_end_time
        self.bag = []
        self.curr_activity = ''
        self.specific_event_requirement = ''

        self.memory = []
        self.embeddings = {}
        self.high_level_plans = []
        self.curr_high_level_index = 0
        # the actual game starts on 2023-01-03, so setting provided info as the day before
        global_start_date = datetime.fromisoformat('2023-01-02')

        self.load(path)
        


        if llm_provider == "openai":
            self.LLM = OpenAILLM(llm_model_name=llm_model_name, embedding_model_name=embedding_model_name)
        # elif llm_provider == "mindsdb":
        #     self.LLM = MindsDBLLM(llm_model_name=llm_model_name, embedding_model_name=embedding_model_name)
        # else:
        #     self.LLM = LocalLLM(llm_model_name=llm_model_name, embedding_model_name=embedding_model_name)
        
        if self.memory == []:
            statements = self.description.split(';') if isinstance(self.description, str) else self.description
            for statement in statements:
                self.add_to_memory(
                    activity=statement.strip(), 
                    curr_time=global_start_date,
                    memory_type="provided_statements"
                )
      
    def __lt__(self, other):
        if self.action_end_time == other.action_end_time:
            return self.name < other.name
        return self.action_end_time < other.action_end_time
    
    def load(self, file_path):
        scratch_path = f"{file_path}/scratch.json" 
        embedding_path = f"{file_path}/embeddings.json"
        scratch = load_json_file(scratch_path)
        self.name = scratch["name"]
        self.description = scratch["description"]
        self.daily_plan_req = scratch["daily_plan_req"]
        self.currently = scratch["currently"]
        self.learned = scratch["learned"]
        self.lifestyle = scratch["lifestyle"]
        self.age = scratch["age"]
        self.traits = scratch["traits"]
        self.curr_tile = scratch["curr_tile"]
        self.social_relationships = scratch["social_relationships"]
        if "action_end_time" in scratch:
            self.action_end_time = datetime.fromisoformat(scratch["action_end_time"])
        if "high_level_plans" in scratch:
            self.high_level_plans = scratch["high_level_plans"]
            self.curr_high_level_index = scratch["curr_high_level_index"]
        if "bag" in scratch:
            self.bag = scratch["bag"]
        if "reflection_frequency" in scratch:
            self.reflection_frequency = scratch["reflection_frequency"]
        if "memory" in scratch:
            self.memory = scratch["memory"]
            for item in self.memory:
                item["creation_time"] = datetime.fromisoformat(item["creation_time"])
                item["last_access_time"] = datetime.fromisoformat(item["last_access_time"])
        
        if os.path.exists(embedding_path):
            self.embeddings = load_json_file(embedding_path)
            
    def save(self, file_path):
        scratch_path = f"{file_path}/scratch.json"
        embedding_path = f"{file_path}/embeddings.json"
        print(self.name)
        print(self.action_end_time)
        print(self.high_level_plans)
        memory = [item.copy() for item in self.memory]
        for item in memory:
            item["creation_time"] = datetime.isoformat(item["creation_time"])
            item["last_access_time"] = datetime.isoformat(item["last_access_time"])
        scratch = {
            "name": self.name,
            "description": self.description,
            "daily_plan_req": self.daily_plan_req,
            "currently": self.currently,
            "learned": self.learned,
            "lifestyle": self.lifestyle,
            "age": self.age,
            "traits": self.traits,
            "curr_tile": self.curr_tile,
            "social_relationships": self.social_relationships,
            "action_end_time": datetime.isoformat(self.action_end_time),
            "high_level_plans": self.high_level_plans,
            "curr_high_level_index": self.curr_high_level_index,
            "bag": self.bag,
            "memory": memory,
            "reflection_frequency": self.reflection_frequency
        }
        write_json_file(scratch, scratch_path)
        write_json_file(self.embeddings, embedding_path)
    
    def get_str_iss(self):
        """
    ISS stands for "identity stable set." This describes the commonset summary
    of this persona -- basically, the bare minimum description of the persona
    that gets used in almost all prompts that need to call on the persona. 

    """
        commonset = ""
        commonset += f"Name: {self.name}\n"
        commonset += f"Age: {self.age}\n"
        commonset += f"Innate traits: {self.traits}\n"
        commonset += f"Learned traits: {self.learned}\n"
        commonset += f"Currently: {self.currently}\n"
        commonset += f"Lifestyle: {self.lifestyle}\n"
        commonset += f"Daily plan requirement: {self.daily_plan_req}\n"
        # specific event requirement
        if self.specific_event_requirement:
            commonset += f"Specific event requirement: {self.specific_event_requirement}\n"
        return commonset

    def add_to_memory(self, activity, curr_time, calculate_importance=True, **kwargs):
        if activity in self.embeddings:
            for item in self.memory:
                if item["activity"] == activity:
                    if curr_time > item["last_access_time"] + timedelta(hours=1):
                        memory_item = {
                            "creation_time": curr_time,
                            "activity": activity,
                            "last_access_time": curr_time,
                            "importance": item["importance"],
                        }
                        for arg in kwargs:
                            memory_item[arg] = kwargs[arg]
                        self.memory.append(memory_item)
                    else:
                        item["last_access_time"] = curr_time
                        for arg in kwargs:
                            item[arg] = kwargs[arg]
                    break
        else:
            memory_item = {
                "creation_time": curr_time, 
                "activity": activity, 
                "last_access_time": curr_time,
                "importance": 5 if not calculate_importance else self.calculate_importance(activity),
            }
            for arg in kwargs:
                memory_item[arg] = kwargs[arg]

            self.memory.append(memory_item)
            self.embeddings[activity] = self.LLM.get_embeddings(activity)

    

    def get_relevance_scores(self, query):
        query_embedding = self.LLM.get_embeddings(query)
        # logging.info(json.dumps([memory_item["activity"] for memory_item in self.memory], indent=4))
        memory_item_embeddings = []
        for memory_item in self.memory:
            if memory_item["activity"] not in self.embeddings:
                print("embedding not found for", memory_item["activity"])
                self.embeddings[memory_item["activity"]] = self.LLM.get_embeddings(memory_item["activity"])
            memory_item_embeddings.append(self.embeddings[memory_item["activity"]])
        scores = cosine_similarity([query_embedding], memory_item_embeddings)[0]
        return scores

    def calculate_recency_score(self, time0, time1):
        if type(time1) == str:
            time1 = datetime.fromisoformat(time1)
        if type(time0) == str:
            time0 = datetime.fromisoformat(time0)
        duration_hours = (time1 - time0).total_seconds() // 3600
        score = 0.99**duration_hours
        return score

    def min_max_scaling(self, scores):
        # if min == max, all scores == 1
        min_score = min(scores)
        max_score = max(scores)
        scaled_scores = [(score-min_score+1e-10) / (max_score-min_score+1e-10) for score in scores]
        return scaled_scores

    def combine_scores(self, relevance_scores, importance_scores, recency_scores, relevance_alpha=1, importance_alpha=1, recency_alpha=1):
        combined_scores = []
        for i in range(len(relevance_scores)):
            combined_score = relevance_scores[i] * relevance_alpha
            combined_score += importance_scores[i] * importance_alpha
            combined_score += recency_scores[i] * recency_alpha
            combined_scores.append(combined_score)
        return combined_scores

    def retrieve_memories(self, query, curr_time, top_n=5, timestamp=False):
        relevance_scores = self.get_relevance_scores(query)
        importance_scores = [memory_item["importance"] for memory_item in self.memory]
        recency_scores = [self.calculate_recency_score(memory_item["last_access_time"], curr_time) for memory_item in self.memory]
        combined_scores = self.combine_scores(
            self.min_max_scaling(relevance_scores), 
            self.min_max_scaling(importance_scores), 
            self.min_max_scaling(recency_scores)
        )

        ordered_data = np.argsort(combined_scores)[::-1]
            
        distinct_activities = set()
        memory_statements = []
        
        for index in ordered_data:
            activity = self.memory[index]['activity']
            if activity not in distinct_activities:
                distinct_activities.add(activity)
                if timestamp:
                    memory_statements.append(DatetimeNL.get_formatted_date_time(self.memory[index]['creation_time']) + ' ' + activity)
                else:
                    memory_statements.append(activity)
            if len(memory_statements) >= top_n:
                break
        return memory_statements
    
    def get_questions_for_reflection(self):
        prompt = "\n".join([memory_item["activity"] for memory_item in self.memory[-100:]])
        prompt += f'''Given only the information above, what are 3 most salient high-level questions we can answer about {self.name} in the statements?
Output the questions only, separated by commas.
Output: 
'''
        questions = self.LLM.get_llm_response(prompt, max_tokens=300)
        question_list = [question for question in questions.split(",")]
        return question_list
    
    def reflect(self, curr_time):
        questions = self.get_questions_for_reflection()
        for question in questions:
            print("---------------------------------- question:",question)
            memories = self.retrieve_memories(question, curr_time, top_n=15)
            # print("---------------------------------- memories:",memories)
            prompt = f"Statements about {self.name}\n"
            index_str_to_memory = {str(i): memories[i] for i in range(len(memories))}
            for i, memory in enumerate(memories):
                prompt += f"{i}. {memory}\n"
            prompt += "What 3 high-level insights can you infer from the above statements? (example format: insight (because of 1, 5, 3))"
            prompt += f'''Here is an example:
1. Lila is my aunt.
2. I closed the door and set off for a trip.
3. Today is a good day.
4. Lila likes to eat honey.
5. I plan to travel to Europe.
6. I tend to drive to travel.

Output: 1. My aunt Lila likes to eat honey. (because of 1, 4)
2. I drive to Europe for a trip. (because of 2, 5, 6)
'''
            prompt += "Please output each insight on a new line, without any explaination."
            prompt += "Output:"
            insights = self.LLM.get_llm_response(prompt)
            print('---------------------------------- insights:',insights)
            # remove the 1. 2. or 3. 
            insights_list = []
            for insight in insights.split("\n"):
                if insight.strip() == "":
                    continue
                insights_list.append(insight[2:])
            for insight in insights_list:
                insight_pair = insight.split("(")
                insight_only, reason = insight_pair
                reason = reason.replace(")", " ")
                source_nodes = [node.strip() for node in reason.replace(' ', ",").split(",") if node.strip().isnumeric()]
                source_memories = [index_str_to_memory[source_node] for source_node in source_nodes]
                self.add_to_memory(
                    activity=insight_only.strip(), 
                    curr_time=curr_time,
                    source_memories=source_memories,
                    memory_type="reflect"
                )
        return insights


    @cache
    def calculate_importance(self, memory_statement):
        #example memory statement -  buying groceries at The Willows Market and Pharmacy
        prompt = f'''
On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following piece of memory.
Output in only the number format.
Memory: {memory_statement}
Rating:'''
        return int(self.LLM.get_llm_response(prompt, max_tokens=1))

    def get_agent_information(self, aspect="core characteristics", curr_time=None):
        memory_query = f"{self.name}'s {aspect}"
        memory_statements = self.retrieve_memories(memory_query, curr_time)
        joined_memory_statements = '\n- '.join(memory_statements)
        prompt = f"""How would one describe {memory_query} given the following statements?\n- {joined_memory_statements}"""
        return self.LLM.get_llm_response(prompt)


    @cache
    def get_agent_summary_description(self, curr_time):
        """
        In our implementation, this summary comprises agents'
        identity information (e.g., name, age, personality), as well as a
        description of their main motivational drivers and statements that
        describes their current occupation and self-assessment.

        This is currently cached using the key curr_time, but can be cached based on the day or hour
        """

        core_characteristics = self.get_agent_information(aspect="core characteristics", curr_time=curr_time)
        current_daily_occupation = self.get_agent_information(aspect="current daily occupation", curr_time=curr_time)
        feelings = self.get_agent_information(aspect="feeling about his recent progress in life", curr_time=curr_time)

        description = f"""
        Name: {self.name} (age: {self.age})
        Innate traits: {', '.join(self.traits)}
        {core_characteristics}
        {current_daily_occupation}
        {feelings}
        """
        return description

    def perceive(self, maze, curr_tile, curr_time):
        curr_arena_path = maze.get_tile_path(curr_tile, "arena")
        nearby_tiles = maze.get_nearby_tiles(curr_tile, vision_r)
        percept_events_set = set()
        # We will order our percept based on the distance, with the closest ones
        # getting priorities.
        percept_events_list = []
        # First, we put all events that are occurring in the nearby tiles into the
        # percept_events_list
        for tile in nearby_tiles:
            tile_details = maze.access_tile(tile)
            if tile_details["events"]:
                if maze.get_tile_path(tile, "arena") == curr_arena_path:
                    # This calculates the distance between the persona's current tile,
                    # and the target tile.
                    dist = math.dist([tile[0], tile[1]],
                                    [curr_tile[0], curr_tile[1]])
                    # Add any relevant events to our temp set/list with the distant info.
                    for event in tile_details["events"]:
                        if self.name not in event[0] and event[0] not in percept_events_set and event[1] < curr_time:
                            percept_events_list += [[dist, event[0]]]
                            percept_events_set.add(event[0])

        # We sort, and perceive only persona.scratch.att_bandwidth of the closest
        # events. If the bandwidth is larger, then it means the persona can perceive
        # more elements within a small area.
        percept_events_list = sorted(percept_events_list, key=itemgetter(0))
        perceived_events = []
        for dist, event in percept_events_list[:att_bandwidth]:
            perceived_events += [event]
        for event in perceived_events:
            self.add_to_memory(event, curr_time, memory_type="percept")
            

    # @cache
    # def get_agent_action_generative(self, curr_time):
    #     #get what an agent is doing at a specific time given a plan (generated externally first and added to memory)
    #     formatted_date_time = DatetimeNL.get_formatted_date_time(curr_time)
    #     query = f"{formatted_date_time}\nIn one sentence (starting with {self.name}), what is {self.name} doing?"
    #     memories = self.retrieve_memories(query, curr_time, top_n=5)
    #     joined_memory_statements = '\n- '.join(memories)
    #     prompt = f"{joined_memory_statements} {query}"
    #     activity = self.LLM.get_llm_response(prompt)
    #     self.add_to_memory(activity=activity, curr_time=curr_time, memory_type="action")
    #     return activity

    @staticmethod
    def remove_formatting_before_time(one_string):
        for i, char in enumerate(one_string):
            if char.isdigit():
                return one_string[i:]
        return ''

    @cache
    def get_summary_of_relevant_context(self, other_agent, other_agent_activity, curr_time):
        # here the agent can directly access the other agent's memory which is buggy (feature of generative agent), 
        # maybe can only see a fraction on shared memory (in improved version) based on what they know about the other agent
        prompt1 = f"What is {self.name}'s relationship with the {other_agent.name}?"
        prompt2 = other_agent_activity
        memories1 = self.retrieve_memories(prompt1, curr_time, top_n=5)
        memories2 = other_agent.retrieve_memories(prompt2, curr_time, top_n=5)
        joined_memory_statements = '\n- '.join(memories1 + memories2)
        prompt = f"Summarize this: {joined_memory_statements}"
        return self.LLM.get_llm_response(prompt)

    @staticmethod
    def parse_reaction_response(response):
        """
        The first sentence should either contain Yes or No (and maybe some additional words). If yes, the second sentence onwards tells of the actual reaction
        """
        response_parts = response.split(".")
        if "yes" in response_parts[0].lower() and len(response_parts) > 1:
            full_response = '.'.join(response_parts[1:])
            return full_response
        return None

    def get_agent_reaction_about_another_agent(self, other_agent, curr_time):
        #TODO: right now the reaction is only to another agent but in the game world, the agent can respond to other objects as well

        # template is changed to ask for only 1 sentence

        self_activity = self.curr_activity
        other_activity = other_agent.curr_activity

        formatted_date_time = DatetimeNL.get_formatted_date_time(curr_time)
        summary_of_relevant_context = self.get_summary_of_relevant_context(other_agent, other_activity, curr_time)
        self_summary = self.get_agent_summary_description(curr_time)
        prompt = f"""
        {self_summary}
        {formatted_date_time}
        {self.name}'s status: {self_activity}
        Observation: {self.name} saw {other_agent.name} {other_activity}
        Summary of relevant context from {self.name}'s memory:
        {summary_of_relevant_context}
        Should {self.name} react to the observation? Please respond with either yes or no. If yes, please also then suggest an appropriate reaction in 1 sentence.
        """
        reaction_raw = self.LLM.get_llm_response(prompt)
        # print(f"Raw reaction response by {self.name}:",reaction_raw)
        reaction_processed = GenerativeAgent.parse_reaction_response(reaction_raw)
        #based on the paper, need to re-plan with every reaction but don't think super helpful here
        # if reaction_processed is not None:
        #     self.plan(curr_time)
        return reaction_processed

    @staticmethod
    def convert_conversation_in_linearized_representation(conversation_history):
        if not conversation_history:
            return ''
        linearized_conversation_history = 'Here is the dialogue history:\n'
        for turn in conversation_history:
            #sometime conversation last turn has no text
            if turn['text'] is not None:
                linearized_conversation_history += f"{turn['name']}: {turn['text']}\n"
        return linearized_conversation_history

    def speak_to_other_agent(self, other_agent, curr_time, reaction=None, conversation_history=[]):
        if reaction is None:
            return None
        
        self_summary = self.get_agent_summary_description(curr_time)
        formatted_date_time = DatetimeNL.get_formatted_date_time(curr_time)
        self_activity = self.curr_activity
        other_activity = other_agent.curr_activity
        summary_of_relevant_context = self.get_summary_of_relevant_context(other_agent, other_activity, curr_time)
        
        if not conversation_history:
            # first turn, use only the intent of the speaker to ground
            background = f"{self.name} hopes to do this: {reaction}"
        else:
            # else continue the conversation
            background = GenerativeAgent.convert_conversation_in_linearized_representation(conversation_history)
        
        #What would he say next to {other_agent.name}? Please respond in a conversational style.
        prompt = f"""
        {self_summary}
        {formatted_date_time}
        {self.name}'s status: {self_activity}
        Observation: {self.name} saw {other_agent.name} {other_activity}
        Summary of relevant context from {self.name}'s memory:
        {summary_of_relevant_context}
        {background}
        What would {self.name} say next to {other_agent.name}?
        {self.name}:"""
        return self.LLM.get_llm_response(prompt)

    def dialogue(self, other_agent, curr_time, max_turns=6):
        
        #if the reaction indicates an interaction between agents, we generate their dialogue.
        #reaction need not necessarily result in dialogue (can be change in self plan)
        conversation_history = []

        while len(conversation_history) < max_turns and (not conversation_history or conversation_history[-1]["reaction"] is not None):
            # self turn
            response_self = self.get_agent_reaction_about_another_agent(other_agent, curr_time)
            speak_self = self.speak_to_other_agent(other_agent, curr_time, reaction=response_self, conversation_history=conversation_history)
    
            conversation_history.append({
                "name": self.name, 
                "text": speak_self, 
                "reaction": response_self
            })

            if conversation_history[-1]["reaction"] is None:
                return conversation_history

            logging.info(json.dumps(conversation_history[-1], indent=4))
            
            # other turn 
            response_other = other_agent.get_agent_reaction_about_another_agent(self, curr_time)
            speak_other = other_agent.speak_to_other_agent(self, curr_time, reaction=response_other, conversation_history=conversation_history)

            conversation_history.append({
                "name": other_agent.name, 
                "text": speak_other, 
                "reaction": response_other
            })
            logging.info(json.dumps(conversation_history[-1], indent=4))
        linearized_conversation_history = GenerativeAgent.convert_conversation_in_linearized_representation(conversation_history)

        #add dialogue to memory of both agents
        self.add_to_memory(linearized_conversation_history, curr_time, memory_type="dialogue")
        other_agent.add_to_memory(linearized_conversation_history, curr_time, memory_type="dialogue")

        return conversation_history
    
    def convert_to_emoji(self, activity):
        prompt = f"Please represent ```{activity}''' using 2 emoji characters"
        return self.LLM.get_llm_response(prompt, max_tokens=8)

    def generate_high_level_plans(self):
        prompt = f'''You will be planning a day for {self.name} based on the personal information.
Your plan must align with the individual's routine, considering their waking up, eating, and sleeping times.
Every plan should be in the format:{self.name} does [activity] from [start time] to [end time].
Here is an example:

Here is Sam Kissin's personal information:
Name: Sam Kissin
Age: 28
Backstory: Raised by a single mother who was always working, Sam was often left to her own devices. She learned to fend for herself at an early age and developed a hard edge. 
Personality: hard-edged, independent, loyal
Currently: Sam is a painter who is preparing for her first solo show. She mostly works from home.
Daily plan requirement: Sam wakes up at 8 am, she is planning to stay at home all day and never go out of her home.

Generate high-level plans for Sam Kissin:
Sam keeps sleeping from 06:00 am to 08:00 am.
Sam wakes up and completes his morning routine from 08:00 am to 09:30 am.
Sam works on her painting from 09:30 am to 01:00 pm.
Sam has lunch and takes a nap from 01:00 pm to 02:00 pm.
Sam works on her painting from 02:00 pm to 06:00 pm.
Sam has dinner and watches TV from 06:00 pm to 08:00 pm.
Sam works on her painting from 08:00 pm to 11:00 pm.
Sam goes to bed from 11:00 pm to 06:00 am.

Here is {self.name}'s personal information:\n{self.get_str_iss()}

Generate high-level plans for {self.name}:
'''
        # print(prompt)
        response = self.LLM.get_llm_response(prompt)
        plan_list = response.split('\n')
        for plan in plan_list:
            if plan == '':
                continue
            self.high_level_plans.append(plan.strip())


#     def get_agent_activity_type_awareness(self, curr_time):
#         prompt = f"Here is {self.name}'s personal information\n{self.get_str_iss()}\n"
#         formatted_date_time = DatetimeNL.get_formatted_date_time(curr_time)
#         prompt += f'''Here are three different categories of activities that {self.name} can engage in:
# (1) Basic Survival Needs: This category includes all activities related to maintaining life and physical well-being, such as eating, drinking, sleeping, personal hygiene, and health management.
# (2) Work and Responsibilities: This category encompasses tasks related to work, study, and household duties. It covers activities like going to work, attending meetings, studying, household chores, daily tasks and responsibilities.
# (3) Social and Leisure Activities: This group includes activities focused on relaxation, entertainment, and social interaction, such as socializing, entertainment, exercise, hobbies, communication, and self-reflection. 
# {formatted_date_time}
# Which aspect should {self.name} engage in at that time?
# Please respond only with either Basic Survival Needs, Work and Responsibilities, or Social and Leisure Activities.'''
#         response = self.LLM.get_llm_response(prompt)
#         return response
        
    
    def generate_action_location(self, activity, curr_time, maze):
        #generate the location of the action
        #currently, we are using the maze to determine the location of the action
        #this can be improved by using the spatial memory to determine the location
        formatted_date_time = DatetimeNL.get_formatted_date_time(curr_time)
        accessible_sector_str = maze.get_str_accessible_sectors()

        '''

        ---
        It is Tuesday Jan 03 2023 06:00 pm,
        Jane Anderson's incoming action: Jane Anderson is going to eating dinner.
        Jane Anderson's current location: Oak Hill College
        Please select the most likely area Jane Anderson will go based on the information. The answer must be one of the following area option.
        Area options:
        Oak Hill College Student Dormatory, The Rose and Crown Pub, Hobbs Cafe, Oak Hill College, Johnson Park, Harvey Oak Supply Store, The Willows Market and Pharmacy
        Reasoning: Jane Anderson is going to eating dinner, and Hobbs Cafe serves foods all the time, so she should go to Hobbs Cafe.
        Answer: Hobbs Cafe
        ---
        It is Tuesday Jan 03 2023 01:00 pm,
        Sam Kim's incoming action: Sam Kim is exercising and taking a walk to relax.
        Sam Kim's current location: Sam Kim's house
        Please select the most likely area Sam Kim will go based on the information. The answer must be one of the following area option.
        Area options:
        Sam Kim's house, The Rose and Crown Pub, Hobbs Cafe, Oak Hill College, Johnson Park, Harvey Oak Supply Store, The Willows Market and Pharmacy
        Reasoning: for exercising and walking, a large place is needed, so he should go to Johnson Park.
        Answer: Johnson Park'''

        fin_accessible_sectors = accessible_sector_str.split(", ")
        accessible_sector_str = ', '.join(fin_accessible_sectors)
        
        retrieved_memory = self.retrieve_memories(activity, curr_time, top_n=5)
        
        prompt_sector = f'''Task: select the most accurate location for the action to take place.
Note: Stay in the current area if the activity can be done there. 
Note: Only go out if the activity needs to take place in another place.
Note: Never go into other people's rooms unless necessary.
Note: In the format of "Reasoning:... Answer: "
Here is {self.name}'s personal information\n{self.get_str_iss()}
{formatted_date_time},
Here is related memory to help you make a decision:
{retrieved_memory}
{self.name}'s incoming action: {activity}
{self.name}'s current location: {maze.access_tile(self.curr_tile)['sector']}
Please select the most likely area {self.name} will go based on the information. The answer must be one of the following area option.
Area options:
{accessible_sector_str}

'''
        # print(prompt_sector)
        # sector = ""
        # while sector not in fin_accessible_sectors:
        response = self.LLM.get_llm_response(prompt_sector)
        print(response)
        sector = response.split("Answer: ")[1].strip()
        prompt_arena = f'''{self.name}'s current action: {activity}
{self.name} is planning to go to {sector} for the action, which has sub-areas:
{maze.get_str_accessible_sector_arenas(f"{sector}")}

Please select the most likely sub-area {self.name} will go to based on the information. Your answer must adhere to the following constraints:
1) only output the sub-area name.
Output: 
'''
        # arena = ""
        # while arena not in maze.get_accessible_sector_arenas(f"{sector}"):
        arena = self.LLM.get_llm_response(prompt_arena)
        
        prompt_game_object = f'''{self.name} is planning to go to {sector}:{arena}, which has following objects.

Objects:
{maze.get_str_accessible_sector_arena_game_objects(f"{sector}", f"{arena}")}

{self.name}'s current action: {activity}


Please select the most likely game object {self.name} will interact with based on the information. Your answer must adhere to the following constraints:

1) only output the object name without any explaination.
2) The object must be in the list of objects in the area.
Output: 
'''
        # object = ""
        # while object not in maze.get_accessible_sector_arena_game_objects(sector, arena):
        object = self.LLM.get_llm_response(prompt_game_object)
        
        if sector == 'Oak Hill College' and arena == 'hallway':
            object = ''
        print(sector, arena, object)
        if object == '':
            return f"{sector}:{arena}"
        return f"{sector}:{arena}:{object}"

    def whether_to_generate_new_high_level_plan(self, curr_time):
        last_high_level_plan = self.high_level_plans[self.curr_high_level_index]
        end_time = last_high_level_plan.split('to')[-1][:-1].strip()
        time_obj = datetime.strptime(end_time, '%I:%M %p').time()
        curr_time = curr_time.time()
        if curr_time >= time_obj:
            self.curr_high_level_index += 1
            return True
        else: return False
        # formatted_date_time = DatetimeNL.get_formatted_date_time(curr_time)
        # query = f"{formatted_date_time}\nShould {self.name} generate new high-level plan?"
        # history = self.high_level_plans
        # history = '\n- '.join(history)
        # prompt = f"Here is {self.name}'s personal information\n{self.get_str_iss()}\nHigh-level plan history:\n{history}\n {query} \nPlease respond with either yes or no."
        # # print(prompt)
        # response = self.LLM.get_llm_response(prompt)
        # # print(response)
        # if 'yes' in response.lower():
        #     return True
        # else:
        #     return False

    def generate_current_activity_based_on_history(self, curr_time):
        history = [memory_item["activity"] for memory_item in self.memory if memory_item["memory_type"] == "action"]
        
        if len(history) == 1:
            history = history[0]
        else:
            history = '\n- '.join(history[-8:])
        high_level_plans = self.high_level_plans
        formatted_date_time = DatetimeNL.get_formatted_date_time(curr_time)
        # activity_type = self.get_agent_activity_type_awareness(curr_time)
        query = f"what should {self.name} do in the next 15 minutes?"
        retrieved_memories = self.retrieve_memories(query, curr_time, top_n=5)
        print(retrieved_memories)
        # retrieved_memories = '\n- '.join(history[-10:])
        iss = self.get_str_iss()
        prompt = f'''Generate the next plan for {self.name}
Your output needs to comply with the following constraints:
(1) Please only contain one action. 
(2) Your response must consist with the personal information, especially waking up, eating and sleeping time.
(3) Please consider the current high-level plan and the recent activity history, it's okay to generate action that isn't strictly following the high-level plan.
(4) Make sure the action is reasonable and feasible in the context of the persona's life, your response should be concrete and diverse.
(5) In the format of "Reasoning: ...\nAnswer: {self.name} is"
Here is several examples:
---
Here is Kelly Bronson's personal information
Name: Kelly Bronson
Age: 35
Backstory: Kelly always wanted to be a teacher, and now she teaches kindergarten. During the week, she dedicates herself to her students.
Personality: sweet, gentle, meticulous
Currently: Kelly is a teacher during the school year. She teaches at the school but works on lesson plans at home.
Daily plan requirement: Kelly awakes at 7 am, she is planning to teach during the morning and work from home in the afternoon.

It is Tuesday Jan 03 2023 06:30 am,
Reasoning: Since Kelly always awakes at 7 am, and now is 6:30 am, so Kelly should keep sleeping until her waking up time. 
Answer: Kelly is sleeping. 
---
It is Tuesday Jan 03 2023 08:30 am,
Reasoning: According to the daily plan requirement of Kelly, she is planning to teach during morning, and it is 8:30 am now, she should look for notebooks. 
Answer: Kelly is looking for notebooks. 
---
It is Tuesday Jan 03 2023 12:30 pm,
Reasoning: It is noon now, and people always eat lunch at noon, so Kelly should eat lunch now. 
Answer: Kelly is eating lunch. 
---
Here is {self.name}'s personal information
{iss}
Here is the {self.name}'s retrieved memories:
{retrieved_memories}

Here is {self.name}'s high-level plan history:
{''.join(high_level_plans[:self.curr_high_level_index])}
The current high-level plan is: {high_level_plans[self.curr_high_level_index]}
Here is {self.name}'s activity history in the last 2 hours:
{history}
{formatted_date_time}
Let's think step by step.
'''
        # print(prompt) 
        activity = self.LLM.get_llm_response(prompt)
        print("---"*10 + '\n'*3)
        print(activity)
        activity = activity.split('Answer: ')[1]
        
        prompt_time = f'''Given an action, determine how long it should cost, must be longer than 5 minutes and shorter than 30 minutes.
Action: {activity}
{formatted_date_time}
Output the time in minutes, only the number:
'''
        time = int(self.LLM.get_llm_response(prompt_time, max_tokens=10))
        return time, activity
        # else:
        #     activity = history[-1].split(' from ' + DatetimeNL.get_time_nl(DatetimeNL.subtract_15_min(curr_time)))[0]
        #     activity = f"{activity} from {DatetimeNL.get_time_nl(curr_time)} to {DatetimeNL.get_time_nl(DatetimeNL.add_15_min(curr_time))}"
        #     return activity

    def operate_surrounding_objects(self, maze, activity):
        #operate the surrounding objects based on the activity
        objects_list = ", ".join(str(item) for item in maze.object_list)
        prompt = f'''Please select the object that Kelly will interact with during the activity Kelly is eating lunch. 
Here is all the objects:
blanket, magazine, canvas, gardening tools, shampoo, table, books, eraser, flowers, notebooks, chalk, menu, balls, 
sheet music, first aid kit, chair, sponge, body wash, glass, pastries, milk, canned food, weight plates, soap, 
cushion, toaster tongs, banana, dish soap, picture frames, clothes, toilet paper, desk, toilet cleaner, bottle opener, pans,
remote control, pens, guitar tuner, bags, controller, monitor, cake, piano bench,
hangers, prescriptions, receipt, towel, lamp, stationery, microphone stand, 
keyboard, microphone, plant, prescription, guitar pick, bread, apple, music stand, pots, grocery bags, 
stool, eggs, register, mouse, decor, bench, pop filter, watering can, pillows, game disc, 
paintbrush, notes, coffee, cash register, medicine, snacks, cushions, dumbbells, pool sticks

Reasoning: Kelly is eating lunch, so she will interact with the objects that are related to eating. She probably want to eat cake for lunch, so the answer is cake.
Answer: cake

Here is {self.name}'s personal information\n{self.get_str_iss()}
Here is {self.name}'s current high-level plan: {self.high_level_plans[-1]}
Please select the object that {self.name} will interact with during the activity {activity}.
Here is all the objects:
{objects_list}
Your answer must be at most one of the objects above, explaining why and then output the name, in the format of "Reasoning: ...\nAnswer: "
Your answer must be in plain text, without any formatting or markdown.
Let's think step by step.'''
        # print(prompt)
        object = ""
        while object not in maze.object_list:
            object = self.LLM.get_llm_response(prompt, max_tokens=500)
            object = object.split("Answer: ")[-1]
            print(object)
        candidate_places = '\n'.join(maze.feature_dict[object])
        prompt_place = f'''Here is {self.name}'s personal information\n{self.get_str_iss()}
Please select the place that {self.name} will pick the required object before the activity {activity}
Here is all the candidate places, every line is a place:
{candidate_places}
Your answer must be one of the places above, only output the full place name without any explaination, for example "Answer: Hobbs Cafe:cafe:cafe counter".
Your answer must be in plain text, without any formatting or markdown.
Answer: 
'''
        place = ""
        attempts = 0
        while place not in maze.feature_dict[object]:
            place = self.LLM.get_llm_response(prompt_place, max_tokens=200)
            place = place.split("Answer: ")[-1]
            print(place)
            attempts += 1
            if attempts >= 3:
                candidate_places = list(maze.feature_dict[object])
                place_embeddings = [self.LLM.get_embeddings(candidate) for candidate in candidate_places]
                query_embedding = self.LLM.get_embeddings(place)
                similarity_scores = cosine_similarity([query_embedding], place_embeddings)[0]
                best_match_index = np.argmax(similarity_scores)
                place = candidate_places[best_match_index]
                print(f"Selected closest match based on cosine similarity: {place}")
                break
        return object, place
        
    def whether_to_talk(self, curr_time, agents):
        if self.talking_frequency:
            self.talking_frequency -= 1
            return False
        history = [memory_item["activity"] for memory_item in self.memory if memory_item["memory_type"] == "action"]
        
        if len(history) == 1:
            history = history[0]
        else:
            history = '\n- '.join(history[-8:])
        formatted_agents = ''
        for agent in agents:
            last_activity = ([""]+[memory_item["activity"] for memory_item in agent.memory if memory_item["memory_type"] == "action"])[-1]
            formatted_agents += f"{agent.name}: {last_activity}\n"
        prompt = f'''Here is the personal information of {self.name}: \n{self.get_str_iss()}\n
Here is the history of activities of {self.name} in the last 2 hours: \n{history}\n
It is {DatetimeNL.get_formatted_date_time(curr_time)}.
Given the following agents and their last activities, should {self.name} talk to any of them?
{formatted_agents}
Please respond with either yes or no. If yes, please also then output the name of chosen agent.
'''
        response = self.LLM.get_llm_response(prompt)
        if 'yes' in response.lower():
            name = response.split("yes")[-1].strip()
            for agent in agents:
                if agent.name in name:
                    return agent
        return False
    
    def check_action_location(self, action, location):
        prompt = f"Is the action {action} at the location {location} suitable? Please respond with either yes or no."
        response = self.LLM.get_llm_response(prompt)
        if 'yes' in response.lower():
            return True
        return False
    
    def move_step(self, maze, path, record, record_time, status):
        self.perceive(maze, path, record_time)
        record[record_time] = record.get(record_time, {})
        curr_status = status.copy()
        curr_status["curr_tile"] = path
        record[record_time][self.name] = curr_status
        maze.add_event_from_tile(status["activity"], record_time, curr_status["curr_tile"])
        return DatetimeNL.add_time_sec(record_time, step_size)
    
    def get_next_plan(self, maze, curr_time, record, max_attempts=5):
        if len(self.high_level_plans) == 0:
            self.generate_high_level_plans()
            print(self.high_level_plans)
        if self.whether_to_generate_new_high_level_plan(curr_time):
            record_time = curr_time
            for item in self.bag:
                potential_path = path_finder(maze.collision_maze, 
                                    self.curr_tile, 
                                    item[1], 
                                    collision_block_id)
                place = f"{maze.access_tile(item[1])['sector']}:{maze.access_tile(item[1])['arena']}"
                status = {
                    "name": self.name,
                    "activity": f"{self.name} is on the way to {place}",
                    "location": f"{place}",
                    "bag": self.bag,
                }
                for path in potential_path:
                    record_time = self.move_step(maze, path, record, record_time, status)
                self.curr_tile = item[1]
                status["activity"] = f"{self.name} is putting back {item[0]}"
                for i in range(object_interact_step):
                    record_time = self.move_step(maze, self.curr_tile, record, record_time, status)
            self.bag = []
            curr_time = record_time
        time, curr_activity = self.generate_current_activity_based_on_history(curr_time)

        self.add_to_memory(activity=curr_activity, curr_time=curr_time, memory_type=f"action")
        return time, curr_activity, curr_time


    def get_status_json(self, curr_time, maze, record, agents, condition):
        
        if condition:
            self.specific_event_requirement = condition
        self.reflection_frequency -= 1
        if self.reflection_frequency == 0:
            self.reflect(curr_time)
            self.reflection_frequency = reflection_frequency
        
        same_location_agents = bucket_agents_by_location(maze, self, agents)
        chosen_agent = self.whether_to_talk(curr_time, same_location_agents)
        if chosen_agent:
            # moving to the agent
            record_time = curr_time
            while record_time in record and chosen_agent.name in record[record_time]:
                del(record[record_time][chosen_agent.name])
                record_time = DatetimeNL.add_time_sec(record_time, step_size)
            if chosen_agent.curr_tile != self.curr_tile:
                position = maze.assign_location(self.curr_tile)
                potential_path = path_finder(maze.collision_maze, 
                                    chosen_agent.curr_tile, 
                                    position, 
                                    collision_block_id)
                record_time = curr_time
                status_walking = {
                    "name": chosen_agent.name,
                    "activity": f"{chosen_agent.name} is on the way to {self.name}",
                    "location": f"{maze.get_tile_path(self.curr_tile, 'arena')}",
                    "bag": chosen_agent.bag,
                }
                status_waiting = {
                    "name": self.name,
                    "activity": f"{self.name} is waiting for {chosen_agent.name}",
                    "location": f"{maze.get_tile_path(self.curr_tile, 'arena')}",
                    "bag": self.bag,
                }
                for path in potential_path:
                    chosen_agent.move_step(maze, path, record, record_time, status_walking)
                    record_time = self.move_step(maze, self.curr_tile, record, record_time, status_waiting)
                maze.release_location(chosen_agent.curr_tile)
                chosen_agent.curr_tile = position
                curr_time = record_time
            record_time = curr_time
            
            conversations = self.dialogue(chosen_agent, curr_time)
            conversation_length = 0
            for turn in conversations:
                if turn['text'] is not None:
                    conversation_length += len(turn['text'])
            talking_time = (conversation_length + talking_speed - 1) // talking_speed
            
            summary_prompt = f'''Summarize the conversation between the two agents in one sectence. 
If the conversation involves an event or invitation, please be clear about the time and location.
{GenerativeAgent.convert_conversation_in_linearized_representation(conversations)}\n\n'''
            summary = self.LLM.get_llm_response(summary_prompt)
        
            
            # print("OKOK")
            record_agent = [[self, chosen_agent], [chosen_agent, self]]
            for initiator, responser in record_agent:
                record_time = curr_time
                status = {
                    "name": initiator.name,
                    "activity": f"{initiator.name} is talking with {responser.name} about: {summary}",
                    "location": f"{maze.get_tile_path(initiator.curr_tile, 'arena')}",
                    "bag": initiator.bag,
                    "conversation": conversations
                }
                for i in range(talking_time):
                    logging.info(f"Conversations at {record_time}")
                    record_time = initiator.move_step(maze, initiator.curr_tile, record, record_time, status)
                initiator.add_to_memory(activity=status["activity"], curr_time=curr_time, memory_type=f"action")
            curr_time = record_time
            self.action_end_time = curr_time
            chosen_agent.action_end_time = curr_time
            self.talking_frequency = talking_frequency
            chosen_agent.talking_frequency = talking_frequency
            self.reflect(curr_time)
            self.reflection_frequency = reflection_frequency
            chosen_agent.reflect(curr_time)
            chosen_agent.reflection_frequency = reflection_frequency
            status_self = {
                "name": self.name,
                "activity": f"{self.name} is talking with {chosen_agent.name} about: {summary}",
                "location": f"{maze.get_tile_path(self.curr_tile, 'arena')}",
                "bag": self.bag,
                "conversation": conversations
            }
            status_object = {
                "name": chosen_agent.name,
                "activity": f"{chosen_agent.name} is talking with {self.name} about: {summary}",
                "location": f"{maze.get_tile_path(self.curr_tile, 'arena')}",
                "bag": chosen_agent.bag,
                "conversation": conversations
            }
            return [status_self, status_object]
        
        time, activity, curr_time = self.get_next_plan(maze, curr_time, record)
        self.curr_activity = activity
        
        
        location = self.generate_action_location(activity, curr_time, maze) 
        
        # self check action and location
        if not self.check_action_location(maze, activity, location):
            location = self.generate_action_location(activity, curr_time, maze)
        
        # pick up the required object
        required_object, place = self.operate_surrounding_objects(maze, activity)
        print(required_object, place)
        
        record_time = curr_time
        
        if required_object not in [i[0] for i in self.bag]:
            
            if len(self.bag) + 1 > max_bag_size:
                item = random.choice(self.bag)
                self.bag.remove(item)
                position = random.choice(maze.gob_feature_dict[item[1]]['position'])
                place_ = f"{maze.access_tile(position)['sector']}:{maze.access_tile(position)['arena']}"
                position = maze.assign_location(position)
                potential_path = path_finder(maze.collision_maze, 
                                    self.curr_tile, 
                                    position, 
                                    collision_block_id)
                if len(potential_path) * step_size > time * 60:
                    time += len(potential_path) // object_interact_step
                status = {
                    "name": self.name,
                    "activity": f"{self.name} is on the way to {place_}",
                    "location": f"{place_}",
                    "bag": self.bag,
                }
                for path in potential_path:
                    record_time = self.move_step(maze, path, record, record_time, status)
                self.release_location(self.curr_tile)
                self.curr_tile = position
                status["activity"] = f"{self.name} is putting back {item[0]}"
                for i in range(object_interact_step):
                    record_time = self.move_step(maze, self.curr_tile, record, record_time, status)
            
            destination = random.choice(list(maze.address_tiles[place]))
            uniqueid = maze.tiles[destination[1]][destination[0]]['uniqueid']
            print(destination)
            destination = maze.assign_location(destination)
            potential_path = path_finder(maze.collision_maze, 
                                    self.curr_tile, 
                                    destination, 
                                    collision_block_id)
            if len(potential_path) * step_size > time * 60:
                time += len(potential_path) // object_interact_step
            status = {
                "name": self.name,
                "activity": f"{self.name} is on the way to {place} to pick up {required_object}",
                "location": f"{place}",
                "bag": self.bag,
            }
            for path in potential_path:
                record_time = self.move_step(maze, path, record, record_time, status)
            maze.release_location(self.curr_tile)
            self.curr_tile = destination
            print(f"destination: {destination}")
            print(maze.tiles[destination[1]][destination[0]])
            status["activity"] = f"{self.name} is picking up {required_object}"
            for i in range(object_interact_step):
                record_time = self.move_step(maze, self.curr_tile, record, record_time, status)
            self.bag.append([required_object,uniqueid])
        
        
        
        
        # activity 
        destination = random.choice(list(maze.address_tiles[f"{location}"]))
        destination = maze.assign_location(destination)
        potential_path = path_finder(maze.collision_maze, 
                                   self.curr_tile, 
                                   destination, 
                                   collision_block_id)
        if len(potential_path) * step_size > time * 60:
            # TODO: if this period can't reach the destination or cost more than half of the assigned time
            time += len(potential_path) // object_interact_step
        status = {
            "name": self.name,
            "activity": f"{self.name} is on the way to {location}",
            "location": f"{location}",
            "bag": self.bag,
        }
        for path in potential_path:
            record_time = self.move_step(maze, path, record, record_time, status)
        maze.release_location(self.curr_tile)
        self.curr_tile = destination
        self.action_end_time = DatetimeNL.add_time(curr_time, time)
        status["activity"] = activity
        while record_time < self.action_end_time:
            record_time = self.move_step(maze, self.curr_tile, record, record_time, status)
        
        status['time'] = time
        
        
        return status
    
    

'''
和GAsetting做比较
high level plan
low
'''