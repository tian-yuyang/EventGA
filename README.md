# Event a
The major author of this project is Yuyang Tian, supervised by [Tianxing He](https://cloudygoose.github.io/). This project is done during internship at Shanghai Qi Zhi Institute. We have a visualization tool available at https://github.com/Just-A-Pie/MiniAgentStudio/ .
The code is developed based on [GitHub - joonspk-research/generative_agents: Generative Agents: Interactive Simulacra of Human Behavoir](https://github.com/joonspk-research/generative_agents) .

## Contents

- [Event a](#event-a)
  - [Contents](#contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Running a simulation:](#running-a-simulation)
  - [Customize the event requirement](#customize-the-event-requirement)
  - [Map file](#map-file)




## Installation


1. clone the EventGA
```
git clone https://github.com/tian-yuyang/EventGA.git
cd EventGA
```

2. setup the environment
```
conda create -n evagent python=3.9
conda activate evagent
pip install -r requirements.txt
```

## Usage


Prerequisites

1. Edit ```\scripts\llm.py```, fill in your API keys.This project supports multiple LLMs, including: Azure, OpenAI and Claude.

2. Edit the simulation arguments in the folder ```\simulation_args```.Specify the parameters for your simulation, such as the scenario or task.

### Running a simulation:

Example
If you want to simulate a philosophy lecture using OpenAI GPT.

```
cd scripts
 ..\simulation_args\run.bat philosophy_lecture
```

Windows users: avoid using "\\\\" in command.

Run the ```post_process.py``` to generate the record file for visualization. 

```
python post_process.py --config_file "../simulation_args/philosophy_lecture/1.json" --record_folder "../generations/philosophy_lecture"
```

After that, you can pass the generated ```records_for_sim.json``` and Map folder path to [our front-end tool](https://github.com/Just-A-Pie/MiniAgentStudio/) to watch the visualization.

## Customize the event requirement

Edit the ```\daily_event\"Your event".json```, to specify the event and agent included.

Create a folder in simulation_args with your simulation json file.

For example:

```json
{"input_folder_name": "../specific_agents/philosophy_lecture",
"output_folder_name": "../generations/philosophy_lecture",
"agent_filenames": ["Isabella Rodriguez", "Klaus Mueller", "Maria Lopez", "Ayesha Khan", 
"Hailey Johnson", "Francisco Lopez", "Eddy Lin", "Mei Lin", "John Lin", "Wolfgang Schulz", "Sam Moore", "Arthur Burton", "Carmen Ortiz", "Ryan Park", "Tamara Taylor"],
"default_agent_config_filename":"default_agent_config.json",
"start_time":"2023-01-03T06:00:00",
"end_time":"2023-01-03T10:00:00",
"llm_provider":"openai",
"llm_model_name":"gpt-4o-mini",
"embedding_model_name":"text-embedding-ada-002",
"daily_events_filename": "../daily_events/philosophy_lecture.json"
}
```

Specify the start time and end time also the LLM.


## Map file

The pre-defined map is located in ```\sampleMap```, you can use our front-end tool https://github.com/Just-A-Pie/MiniAgentStudio/ to edit the map and visualize the simulation.