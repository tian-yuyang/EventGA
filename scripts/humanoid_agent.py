import json
import random
from functools import cache
import logging

from generative_agent import GenerativeAgent
from utils import DatetimeNL



class HumanoidAgent(GenerativeAgent):

    def __init__(self, name: str, description: str, age: int, traits: list, example_day_plan: str, social_relationships={}, basic_needs=None, emotion=None, llm_provider=None, llm_model_name=None, embedding_model_name=None):
        super().__init__(name, description, age, traits, example_day_plan, llm_provider=llm_provider, llm_model_name=llm_model_name, embedding_model_name=embedding_model_name)

        self.allow_emotion_changes = True if emotion is None else False

        self.emotion = "neutral" if emotion is None else emotion 

        allowed_emotions_list = ["disgusted", "afraid", "sad", "surprised", "happy", "angry", "neutral"]

        if self.emotion not in allowed_emotions_list:
            raise ValueError(f"Emotion {self.emotion} is not in accepted emotion list {allowed_emotions_list}")

        self.basic_needs_config = basic_needs

        self.basic_needs = {}
        for basic_need in basic_needs:
            self.basic_needs[basic_need['name']] = basic_need['start_value']
        
        self.agent_states_nl = self.get_agent_states_nl()

        # this can be dynamically updated based on activities people do with each other
        self.social_relationships = social_relationships 
        self.suggested_changes = []

    def get_agent_states_nl(self):
        agent_states = ""
        
        #basic needs
        number_to_modifier = {
            3: "slightly ",
            2: "",
            1: "very ",
            0: "extremely ",
        }

        for basic_need in self.basic_needs_config:
            attr = basic_need['name']
            unsatisfied_adjective = basic_need["unsatisfied_adjective"]
            if self.basic_needs[attr] < 4:
                modifier = number_to_modifier[self.basic_needs[attr]]
                agent_states += f"{self.name} is {modifier}{unsatisfied_adjective}. "

        #emotions
        if self.emotion != "neutral":
            agent_states += f"{self.name} is feeling extremely {self.emotion}. "
        return agent_states

    @cache
    def get_summary_of_relevant_context(self, other_agent, other_agent_activity, curr_time):

        # Generative agent can directly access the other agent's memory which is buggy, 
        # here, we improve this by only access agent's own memory
        # this way, each agent can only see a fraction on shared memory based on what they know about the other agent

        prompt1 = f"What is {self.name}’s relationship with the {other_agent.name}?"
        prompt2 = f"What does {self.name} know about {other_agent.name} regarding {other_agent_activity}?"

        memories1 = self.retrieve_memories(prompt1, curr_time, top_n=5)
        memories2 = self.retrieve_memories(prompt2, curr_time, top_n=5)

        joined_memory_statements = '\n- '.join(memories1 + memories2)
        prompt = f"Summarize this: {joined_memory_statements}"
        return self.LLM.get_llm_response(prompt)

    @cache
    def get_agent_action_generative(self, curr_time):
        #get what an agent is doing at a specific time given a plan (generated externally first and added to memory)

        agent_states_nl = self.get_agent_states_nl()

        formatted_date_time = DatetimeNL.get_formatted_date_time(curr_time)
        query = f"{formatted_date_time}\n {agent_states_nl}In one sentence, what is {self.name} doing?" # (starting with {self.name})
        memories = self.retrieve_memories(query, curr_time, top_n=5, timestamp=True)
        #print(memories)
        joined_memory_statements = '\n- '.join(memories)
        prompt = f"{joined_memory_statements} {query}"
        activity = self.LLM.get_llm_response(prompt)
        self.analyze_agent_activity(activity)
        self.add_to_memory(activity=activity, curr_time=curr_time, memory_type="action")
        return activity

    @staticmethod
    def change_plans_helper(LLM, suggested_change, existing_plan):
        prompt = f"""
Please use the suggested change ({suggested_change}) to edit activities in the original plan.
Format: hh:mm am/pm: <activity within 10 words>

original plan: 
{existing_plan}

updated plan:
"""
        llm_response = LLM.get_llm_response(prompt)
        plan = HumanoidAgent.postprocess_change_plans_helper(llm_response)
        return plan

    @staticmethod
    def postprocess_change_plans_helper(plan):
        plan = plan.split('\n')
        plan = '\n'.join([plan_item for plan_item in plan if plan_item.strip()])
        return plan
    
    def change_plans(self, suggested_change, curr_time, max_attempts=10):
        existing_plan = self.get_plan_after_curr_time(curr_time)
        time_nl = DatetimeNL.get_time_nl(curr_time)
        plan = None
        attempts = 0
        while not GenerativeAgent.check_plan_format(plan) and attempts < max_attempts:
            plan = HumanoidAgent.change_plans_helper(self.LLM, suggested_change, existing_plan)

            attempts += 1
            logging.info(f"replanning at {time_nl} attempt number {attempts} / {max_attempts}")
            if not GenerativeAgent.check_plan_format(plan):
                logging.info("Failed Plan")
            logging.info(plan)

        if attempts == max_attempts:
            logging.info("Existing plan")
            logging.info(existing_plan)
            return None
        return plan

    @cache
    def get_agent_action_retrieval_based(self, curr_time):
        planned_activities = self.get_plan_after_curr_time(curr_time)
        agent_states_nl = self.get_agent_states_nl()

        # if there's an emotional/basic needs concern that makes agent inclined to change plan
        if agent_states_nl:
            formatted_date_time = DatetimeNL.get_formatted_date_time(curr_time)
            prompt = f"""
            {formatted_date_time}
            Original plan: {planned_activities}
            Feelings: {agent_states_nl}
            Should {self.name} change their original plan? Please respond with either yes or no. If yes, please also then suggest a specific change in 1 sentence.
            """
            reaction_raw = self.LLM.get_llm_response(prompt)
            suggested_change = GenerativeAgent.parse_reaction_response(reaction_raw)
            self.suggested_changes.append((suggested_change, curr_time))

            if suggested_change is not None:
                updated_plan = self.change_plans(suggested_change, curr_time)
                if updated_plan is not None:
                    self.add_to_memory(activity=updated_plan, curr_time=curr_time, memory_type="15 minutes plan")
                # print("plans changed")
                # print("original plans")
                # print(planned_activities)
                # print("new plans")
                # print(updated_plan)
            
        activity = self.get_agent_action_retrieval_only(curr_time)
        self.analyze_agent_activity(activity)
        return activity

    @staticmethod
    def postprocess_get_emotion_about_activity(emotion, possible_emotions):
        for possible_emotion in possible_emotions:
            if possible_emotion in emotion.lower():
                return possible_emotion
        return possible_emotions[0]

    def get_emotion_about_activity(self, activity):
        possible_emotions = ['neutral', 'disgusted', "afraid", "sad", "surprised", "happy", "angry"]

        emotion_prompt = f"In the following activity '{activity}', what emotion is expressed? Please reply in one word from this list {possible_emotions} only."

        llm_response =  self.LLM.get_llm_response(emotion_prompt, max_tokens=4)

        return HumanoidAgent.postprocess_get_emotion_about_activity(llm_response, possible_emotions)

    def analyze_agent_activity(self, activity):
    
        # emotions
        if self.allow_emotion_changes:
            self.emotion = self.get_emotion_about_activity(activity)
    
        # basic needs
        for basic_need in self.basic_needs_config:
            attr = basic_need['name']
            action = basic_need['action']
            decline_likelihood_per_time_step = basic_need["decline_likelihood_per_time_step"]

            
            prompt = f"In normal settings, does the activity '{activity}' involve {action}? Please respond with either yes or no."
            response = self.LLM.get_llm_response(prompt, max_tokens=2)
            
            if 'yes' in response.lower():
                self.__dict__["basic_needs"][attr] += 1
            else:
                if random.random() < decline_likelihood_per_time_step:
                    self.__dict__["basic_needs"][attr] -= 1

            # set floor to 0 and ceiling to 10 for all basic_needs
            self.__dict__["basic_needs"][attr] = min(10, self.__dict__["basic_needs"][attr])
            self.__dict__["basic_needs"][attr] = max(0, self.__dict__["basic_needs"][attr])

        return None
    

    def get_sentiment_about_conversation(self, linearized_conversation_history, other_agent):
        prompt = f"Given this conversation ```{linearized_conversation_history}''' Did {self.name} enjoy the conversation? Please respond with either yes or no."
        response = self.LLM.get_llm_response(prompt, max_tokens=3)
        self.social_relationships[other_agent.name]['closeness'] += 1 if 'yes' in response.lower() else -1

        if self.allow_emotion_changes:
            self.emotion = self.get_emotion_about_activity(linearized_conversation_history)
        return None

    def dialogue(self, other_agent, curr_time, max_turns=10):
        #if the reaction indicates an interaction between agents, we generate their dialogue.
        #reaction need not necessarily result in dialogue (can be change in self plan)
        conversation_history = []

        while len(conversation_history) < max_turns and (not conversation_history or conversation_history[-1]["reaction"] is not None):
            # self turn

            # if first turn
            response_self = self.get_agent_reaction_about_another_agent(other_agent, curr_time, conversation_history=conversation_history)
            
            print("response_self", response_self)

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
            response_other = other_agent.get_agent_reaction_about_another_agent(self, curr_time, conversation_history=conversation_history)

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

        #get sentiment (i.e increase or decrease closeness given conversation)
        linearized_conversation_history = GenerativeAgent.convert_conversation_in_linearized_representation(conversation_history)
        self.get_sentiment_about_conversation(linearized_conversation_history, other_agent)
        other_agent.get_sentiment_about_conversation(linearized_conversation_history, self)

        return conversation_history
    
    def get_status_json(self, curr_time, world_location):
        self.get_15m_plan(curr_time)

        activity = self.get_agent_action_retrieval_based(curr_time)
        activity_emoji = self.convert_to_emoji(activity)
        location = self.get_agent_location(activity, curr_time, world_location)
        most_recent_15m_plan = [memory_item for memory_item in self.memory if memory_item['memory_type'] == '15 minutes plan'][-1]['activity']
        most_recent_day_plan = [memory_item for memory_item in self.memory if memory_item['memory_type'] == 'day_plan'][-1]['activity']

        # tracking most recent suggested change to understand suggestion made to change
        most_recent_suggested_change = self.suggested_changes[-1] if self.suggested_changes else None
        if most_recent_suggested_change is not None:
            changed_content, changed_time = most_recent_suggested_change
            # suggestion don't at curr_time
            if changed_time == curr_time:
                most_recent_suggested_change = changed_content
            else:
                most_recent_suggested_change = None

        status = {
            "name": self.name,
            "activity": activity,
            "activity_emoji": activity_emoji,
            "most_recent_15m_plan": most_recent_15m_plan,
            "most_recent_day_plan": most_recent_day_plan,
            "most_recent_suggested_change": most_recent_suggested_change,
            "location": location, 
            "emotion": self.emotion,
            "basic_needs": self.basic_needs,
            "social_relationships": self.social_relationships
        }
        return status
    

    def speak_to_other_agent(self, other_agent, curr_time, reaction=None, conversation_history=[]):
        # traits are already part of self.summary
        # for Humanoid agent, we added the emotion/emotion/basic needs/closeness on top of Generative Agent

        if reaction is None:
            return None
        
        self_summary = self.get_agent_summary_description(curr_time)
        formatted_date_time = DatetimeNL.get_formatted_date_time(curr_time)
        self_activity = self.get_agent_action_retrieval_only(curr_time)
        other_activity = other_agent.get_agent_action_retrieval_only(curr_time)
        
        basic_needs_and_emotional_state = self.get_agent_states_nl()
        closeness = self.get_closeness_between_self_and_other_agent(other_agent)
        summary_of_relevant_context = self.get_summary_of_relevant_context(other_agent, other_activity, curr_time)

        linearized_conversation_history = GenerativeAgent.convert_conversation_in_linearized_representation(conversation_history) 
        
        self_name = self.name 
        other_name = other_agent.name
        return HumanoidAgent.speak_to_other_agent_helper(self.LLM, self_summary, basic_needs_and_emotional_state, closeness, formatted_date_time, self_name, self_activity, other_name, other_activity, summary_of_relevant_context, reaction, linearized_conversation_history)


    @staticmethod
    def speak_to_other_agent_helper(LLM, self_summary, basic_needs_and_emotional_state, closeness, formatted_date_time, self_name, self_activity, other_name, other_activity, summary_of_relevant_context, reaction, linearized_conversation_history):
        prompt = f"""
{self_summary}
{"Feelings: " + basic_needs_and_emotional_state if basic_needs_and_emotional_state else ""}
Closeness: {closeness}
{formatted_date_time}
{self_name}’s status: {self_activity}
Observation: {self_name} saw {other_name} {other_activity}
Summary of relevant context from {self_name}’s memory:
{summary_of_relevant_context}
{self_name} hopes to do this: {reaction}
{linearized_conversation_history}
What would {self_name} say next to {other_name}?
{self_name}:"""
        llm_response = LLM.get_llm_response(prompt)

        # in case there are leading or trailing whitespace

        llm_response = llm_response.strip()
        # in case there are multiple alternative responses presented, use the first one
        if '\n' in llm_response:
            llm_response = llm_response.split('\n')[0]
        
        # in case response starts and ends with "" as a substring within the text (i.e. sometime includes some reason on why it's said this way)
        if len(llm_response.split('"')) == 3:
            llm_response = llm_response.split('"')[1]

        return llm_response

    def get_closeness_between_self_and_other_agent(self, other_agent):
        # note this value is not symmetrical 
        # ie self.get_closeness_between_self_and_other_agent(other_agent) != other_agent.get_closeness_between_self_and_other_agent(self)
        
        closeness_value = self.social_relationships[other_agent.name]['closeness']

        if closeness_value < 5:
            description = "distant"
        elif closeness_value < 10:
            description = "rather close"
        elif closeness_value < 15:
            description = "close"
        else:
            description = "very close"

        closeness_description = f"{self.name} is feeling {description} to {other_agent.name}"
        return closeness_description

    @staticmethod
    def get_agent_boolean_reaction_about_another_agent_helper(LLM, self_summary, basic_needs_and_emotional_state, closeness, formatted_date_time, self_name, self_activity, linearized_conversation_history, other_name, other_activity, summary_of_relevant_context):
        yes_or_no_prompt = f"""
{self_summary}
{"Feelings: " + basic_needs_and_emotional_state if basic_needs_and_emotional_state else ""}
Closeness: {closeness}
{formatted_date_time}
{self_name}’s status: {self_activity}
Observation: {self_name} saw {other_name} {other_activity}
{linearized_conversation_history}
Summary of relevant context from {self_name}’s memory:
{summary_of_relevant_context}
Should {self_name} react to the observation? Please respond with only yes or no.
""" 
        llm_response = LLM.get_llm_response(yes_or_no_prompt, max_tokens=2)
        return "yes" in llm_response.lower()

    @staticmethod
    def get_agent_reaction_about_another_agent_helper(LLM, self_summary, basic_needs_and_emotional_state, closeness, formatted_date_time, self_name, self_activity, linearized_conversation_history, other_name, other_activity, summary_of_relevant_context):
        reaction_prompt = f"""
{self_summary}
{"Feelings: " + basic_needs_and_emotional_state if basic_needs_and_emotional_state else ""}
Closeness: {closeness}
{formatted_date_time}
{self_name}’s status: {self_activity}
Observation: {self_name} saw {other_name} {other_activity}
{linearized_conversation_history}
Summary of relevant context from {self_name}’s memory:
{summary_of_relevant_context}
In 1 sentence, how should {self_name} react to the observation?
"""
        llm_response = LLM.get_llm_response(reaction_prompt)
        return llm_response.strip()


    def get_agent_reaction_about_another_agent(self, other_agent, curr_time, conversation_history=None):
        # for Humanoid agent, we added the emotion/basic needs and closeness on top of Generative Agent
        # later on we can do ablation of emotion/basic needs/closeness in conversation
        self_activity = self.get_agent_action_retrieval_only(curr_time)
        other_activity = other_agent.get_agent_action_retrieval_only(curr_time)
        basic_needs_and_emotional_state = self.get_agent_states_nl()
        closeness = self.get_closeness_between_self_and_other_agent(other_agent)

        formatted_date_time = DatetimeNL.get_formatted_date_time(curr_time)
        summary_of_relevant_context = self.get_summary_of_relevant_context(other_agent, other_activity, curr_time)
        self_summary = self.get_agent_summary_description(curr_time)

        self_name = self.name
        other_name = other_agent.name
        linearized_conversation_history = GenerativeAgent.convert_conversation_in_linearized_representation(conversation_history)

        react_or_not = HumanoidAgent.get_agent_boolean_reaction_about_another_agent_helper(self.LLM, self_summary, basic_needs_and_emotional_state, closeness, formatted_date_time, self_name, self_activity, linearized_conversation_history, other_name, other_activity, summary_of_relevant_context)

        if react_or_not:
            return HumanoidAgent.get_agent_reaction_about_another_agent_helper(self.LLM, self_summary, basic_needs_and_emotional_state, closeness, formatted_date_time, self_name, self_activity, linearized_conversation_history, other_name, other_activity, summary_of_relevant_context)
        else: 
            return None

        #based on Generative Agents paper, need to re-plan with every reaction but don't think super helpful here
        # if reaction_processed is not None:
        #     self.plan(curr_time)
        #return reaction_processed


