import json
import random
import yaml

from collections import defaultdict
from datetime import datetime, timedelta

env_matrix = "../miniAgent_sampleMap_new"
reflection_frequency = 6
collision_block_id = '32125'
step_size = 5
object_interact_step = 6
vision_r = 6
att_bandwidth = 8
talking_frequency = 2
talking_speed = 50
max_bag_size = 8

def load_json_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json_file(data, filename):
    with open(filename, "w", encoding="utf-8") as fw:
        fw.write(json.dumps(data, indent=4))

def bucket_agents_by_location(maze, agent, agents):
    location_to_agents = []
    location = f"{maze.access_tile(agent.curr_tile)['sector']}:\
{maze.access_tile(agent.curr_tile)['arena']}"
    for other_agent in agents:
        agent_location = f"{maze.access_tile(other_agent.curr_tile)['sector']}:\
{maze.access_tile(other_agent.curr_tile)['arena']}"
        # to make agent location hashable
        if agent_location == location:
            location_to_agents.append(other_agent)
    return location_to_agents

def get_pairwise_conversation_by_agents_in_same_location(agent, location_to_agents, curr_time):
    # only 1 conversation per location, when there are 2 or more agents

    responder = random.sample(location_to_agents, 1)
    convo_history = agent.dialogue(responder, curr_time)
    return convo_history

def get_curr_time_to_daily_event(daily_events_filename):
    curr_time_to_daily_event = defaultdict(None)

    if daily_events_filename is not None:
        with open(daily_events_filename, 'r') as file:
            loaded_json = json.load(file)
        for date, event in loaded_json.items():
            time = "12:00 am"
            curr_time = DatetimeNL.convert_nl_datetime_to_datetime(date, time)
            curr_time_to_daily_event[curr_time.strftime("%y-%m-%d")] = event
    return curr_time_to_daily_event


class DatetimeNL:

    @staticmethod
    def get_date_nl(curr_time):
        # e.g. Monday Jan 02 2023
        day_of_week = curr_time.strftime('%A')
        month_date_year = curr_time.strftime("%b %d %Y")
        date = f"{day_of_week} {month_date_year}"
        return date

    @staticmethod
    def get_time_nl(curr_time):
        #e.g. 12:00 am and 07:00 pm (note there is a leading zero for 7pm)
        time = curr_time.strftime('%I:%M:%S %p').lower()
        return time

    @staticmethod
    def convert_nl_datetime_to_datetime(date, time):
        # missing 0 in front of time
        if len(time) != len("12:00 am"):
            time = "0" + time.upper()
        
        concatenated_date_time = date + ' ' + time
        curr_time = datetime.strptime(concatenated_date_time, "%A %b %d %Y %I:%M %p")
        return curr_time

    def subtract_15_min(curr_time):
        return curr_time - timedelta(minutes=15)
    
    def add_15_min(curr_time):
        return curr_time + timedelta(minutes=15)
    def add_time(curr_time, time):
        return curr_time + timedelta(minutes=time)
    def add_time_sec(curr_time, time):
        return curr_time + timedelta(seconds=time)
        
    @staticmethod
    def get_formatted_date_time(curr_time):
        # e.g. "It is Monday Jan 02 2023 12:00 am"
        date_in_nl = DatetimeNL.get_date_nl(curr_time)
        time_in_nl = DatetimeNL.get_time_nl(curr_time)
        formatted_date_time = f"It is {date_in_nl} {time_in_nl}"
        return formatted_date_time
    
    @staticmethod
    def get_date_range(start_date, end_date):
        """
        Get date range between start_date (inclusive) and end_date (inclusive)
        
        start_date and end_date are str in the format YYYY-MM-DD
        """
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        date_range = []
        
        while start_date <= end_date:
            date_range.append(start_date.strftime('%Y-%m-%d'))
            start_date += timedelta(days=1)
        if not date_range:
            raise ValueError("end_date must be later or equal to start_date")
        return date_range
