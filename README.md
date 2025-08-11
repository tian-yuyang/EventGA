## Event agent


## Contents

- [Event agent](#event-agent)
- [Contents](#contents)
- [Installation](#installation)
- [Get Started](#get-started)
- [Customizing locations and specific agents](#customizing-locations-and-specific-agents)
- [Analytics Dashboard](#analytics-dashboard)
- [Unity WebGL Game interface](#unity-webgl-game-interface)
- [How does Humanoid Agents work?](#how-does-humanoid-agents-work)
- [(NEW) Server-client mode](#new-server-client-mode)
- [(Optional) Adding new basic needs](#optional-adding-new-basic-needs)
- [(Optional/Advanced) Extending HumanoidAgent class](#optionaladvanced-extending-humanoidagent-class)
- [Future Plans](#future-plans)
- [Citation](#citation)




## Installation

```
git clone https://github.com/tian-yuyang/EventGA.git
cd EventGA
conda create -n evagent python=3.9
conda activate evagent
pip install -r requirements.txt
```


## Get Started

Run a simulation like this:

```
cd humanoidagents
python run_simulation_server.py --output_folder_name ../generations/big_bang_theory \
--map_filename ../locations/big_bang_map.yaml \
--agent_filenames ../specific_agents/sheldon_cooper.json ../specific_agents/leonard_hofstadter.json ../specific_agents/penny.json
```

Windows users: avoid using "\\" in command

Prerequisites

1. ```export OPENAI_API_KEY=sk-...``` to use your OpenAI Key if you would like to use OpenAI LLM; otherwise please select ```--llm_provider local``` to run local inferencing models instead and ```export MINDSDB_API_KEY=sk-...``` to use gpt-3.5-turbo through MindsDB instead. Please be careful not to exceed your quota since every simulated day with 2 to 3 agents cost around $2-5 and takes 45-60 minutes given the number of API calls. For **Windows** users, this should be ```set OPENAI_API_KEY=sk-...```

Required arguments

1. ```--output_folder_name``` refers to the folder where the generated output will be stored
2. ```--map_filename``` refers to the filename of the map used (see section below for list of built in maps)
3. ```--agent_filenames``` refers to the list of agent specifications (see section below for list of built in agents)

Optional arguments

1. ```--default_agent_config_filename``` refers to the default agent config file where we define the types of basic needs that every agent has. For more details, refer to the section below.

2. ```--start_date``` refers to the (inclusive) start date of the interested date range. The format is YYYY-MM-DD e.g. 2023-01-03. Kindly note that the date should not be earlier than 2023-01-03 since we use 2023-01-02 as the global_start_date for user-defined memories. If that is required, please adjust the global start date in code.

3. ```--end_date``` refers to the (inclusive) end date  of the interested date range. The format is YYYY-MM-DD e.g. 2023-01-04
   
4. ```--condition``` as noted in the paper, we can adjust the starting condition of all agents (in terms of their basic needs, emotion and closeness to others). You can use this to specify a condition (e.g. health) for all agents to be 0. See the list of accepted arguments on argparse

5. ```--llm_provider```  refers to the Large Language Model provider you would to use. Choose between 
 - ```local```  for a locally hosted LLM (such as Mistral 7B, Mixtral or any LlaMA models) and a local embedding model (such as sentence-transformers/all-MiniLM-L6-v2).  For ```local```, you would also need to start a OpenAI-compatible server. There are many ways to do this but we recommend [LM Studio](https://lmstudio.ai/), a no-code solution equipped with a GUI, as a first attempt to do this.
 - ```openai``` (default) for ChatGPT-3.5-turbo for LLM (by default and configurable to other models) and Ada-v2 (by default and configurable to other models) for embedding respectively. Please note that the openai option charges to yout OpenAI account and you would need to set ```export OPENAI_API_KEY```
 - ```mindsdb``` for ChatGPT-3.5-turbo for LLM through MindsDB. Please note that since MindsDB does not come with embedding model support, this will use OpenAI for embedding directly and hence you would still need to set the ```export OPENAI_API_KEY=sk-...``` in addition to ```export MINDSDB_API_KEY=sk-...``` 

6. ```--llm_model_name``` refers to LLM model name you would like to use.

- for ```--llm_provider=local```: this field does not influence the model being served but feel free to note down the name of model for record-keeping/later analysis

- for ```--llm_provider=openai```: this can any model that's compatible with the chat_completion endpoint (more at https://platform.openai.com/docs/models) - we recommend starting with  ```gpt-3.5-turbo``` (default) or ```gpt-4o```

- for ```--llm_provider=mindsdb```, this can be any model from https://docs.mdb.ai/docs/models - we recommend starting with  ```gpt-3.5-turbo``` (default)

7. ```--embedding_model_name``` refers to Embedding model name you would like to use.

- for ```--llm_provider=local```: please use any model compatible with SentenceTransformers. We recommend starting with ```all-MiniLM-L6-v2```

- for ```--llm_provider=openai``` : please use any model compatible with the embeddings endpoint. We recommend starting with ```text-embedding-ada-002``` (default)

- for ```--llm_provider=mindsdb```, please use the same options as ```--llm_provider=openai```, since MindsDB does not have good embedding model support yet, these embedding are routed to OpenAI directly.


8. ```--daily_events_filename``` refers to major events affecting all agents in a simulation, to provide simulation based on customized settings of your preference. For an example of the expected structure, see ```daily_events/example.yaml``` 

## Customizing locations and specific agents

Currently, we support three built-in settings

1. **Big Bang Theory** ```--map_filename ../locations/big_bang_map.yaml \
--agent_filenames ../specific_agents/sheldon_cooper.json ../specific_agents/leonard_hofstadter.json ../specific_agents/penny.json```

2. **Friends** ```--map_filename ../locations/friends_map.yaml \
--agent_filenames ../specific_agents/joey_tribbiani.json ../specific_agents/monica_gellor.json ../specific_agents/rachel_greene.json```

3. **Lin Family** ```--map_filename ../locations/lin_family_map.yaml \
--agent_filenames ../specific_agents/eddy_lin.json ../specific_agents/john_lin.json```

To create your own setting, you can create your own map as well as your own specific agents. The fields you need to fill for each can be learned from looking at the examples. 

One thing to know is that agents and map are not completely decoupled. For every agent_filename you specify, the name field of the agent has to be contained in the map.yaml under Agents as a key. This sets the initial location of the agent on the map. 

## Analytics Dashboard

![image info](img/analytics_dashboard.png)

The generated data can be visualized by a interactive dashboard. You can select the agent in the world to visualize their status. 

It consists of the graph of basic needs and the graph of social relationship with the corresponding information including the emotion, conversation details.

To run the dashboard, run the following
```
cd humanoidagents
python run_dashboard.py --folder <folder/containing/generation/output/from/run_simulation.py> 
```

Required arguments

1. ```--folder``` refers to the folder where the generated output have be stored from run_simulation.py
2. ```--mode``` refers to the method of selecting data from the folder. It has two modes: 1) ```all```: visualizing all files in the folder 2) ```date_range```: visualizing files with interested date range (need to state the date range in arguments)

Optional arguments
1. ```--start_date``` refers to the (inclusive) start date  of the interested date range when ```--mode = date_range```. The format is YYYY-MM-DD e.g. 2023-01-03
2. ```--end_date``` refers to the (inclusive) end date  of the interested date range when ```--mode = date_range```. The format is YYYY-MM-DD e.g. 2023-01-04

## Unity WebGL Game interface

The Game Interface using Unity WebGL is available on [humanoidagents.com](https://www.humanoidagents.com/)

Support for customized locations and agents is coming soon!

See a 2 minute YouTube Walkthrough below. 

[![Video](https://img.youtube.com/vi/vQkOf-zS2Y0/maxresdefault.jpg)](https://www.youtube.com/watch?v=vQkOf-zS2Y0)

## How does Humanoid Agents work?

![image info](img/system_architecture.png)

**Step 1.** Agent is initialized based on user-provided seed information. 
**Step 2.** Agent plans their day.  
**Step 3.** Agent takes an action based on their plan. 
**Step 3a.** Agent can converse with another agent if in the same location, which can affect the closeness of their relationship.
**Step 4.** Agent evaluates if action taken changes their basic needs status and emotion. 
**Step 5.** Agent can update their future plan based on the satisfaction of their basic needs and emotion. 

## (NEW) Server-client mode

The standard approach of using `run_simulation.py` runs a simulation locally and saves all of the generated files so that they can be loaded into our analytics dashboard and Unity WebGL Game interface. 

We recently discovered that there are certain use-cases that can benefit from real-time simulation of humanoid agents and hence developed a Flask-based REST API to interact with Humanoid Agents. To use this, simply start a server by replacing `run_simulation.py` with `run_simulation_server.py` in [Get Started](#Get-Started), which supports all of the same features].

Then on your client side, do

0. If you're starting a local server, `<BASE_URL>` should be `http://127.0.0.1:5000`
1. Visit `<BASE_URL>/plan?curr_date=2023-01-03` at the start of each simulated day. This plans the day for each agent.
2. Visit `<BASE_URL>/logs?curr_date=2023-01-03&specific_time=09:00` every 15 minutes, replace `09:00` with the time in `hh:mm` format.
3. (optional) Under the hood, `<BASE_URL>/logs` actually calls `<BASE_URL>/activity` and `<BASE_URL>/conversations`, which identifies the activity (at 15 minute interval) and the conversations between agents at each location.
4. (optional) `<BASE_URL>/activity` and `<BASE_URL>/plan` can also be done for each agent individually. This can be done by visiting <BASE_URL>/activity_single and <BASE_URL>/plan_single respectively, with the additional argument of ```name=<agent_name>```. If you are testing this in your browser, be sure to replace a space with `%20` (as in `John Lin` to `John%20Lin`)
5. (optional) Each method currently supports both GET and POST requests for ease of testing in a browser. However, this cannot be guaranteed in the future given the limitations of GET requests and we would recommend POST requests (with data sent under the json param) for future proofness.


## (Optional) Adding new basic needs

You might also be interested to add/remove further basic needs to agents other than the five we have as a default (fullness, social, health, fun and energy)

To do that, you can create your own `default_agent_config.json` file.

Each basic need requires the following format in order for the code to support them.

```json
{
    "name": "fullness", 
    "start_value": 5, 
    "unsatisfied_adjective": "hungry", 
    "action": "eating food", 
    "decline_likelihood_per_time_step": 0.05, 
    "help": "from 0 to 10, 0 is most hungry; increases or decreases by 1 at each time step based on activity"
}
```

## (Optional/Advanced) Extending HumanoidAgent class

If you're reading this, you have played around with the code and now you're ready to take it to the next level.

Instead of using our code, you want to extend it to support more aspects for an Agent such as personality, empathy, moral values or whatever aspect you're interested in.

We provide an abstract interface at ```customized_humanoid_agent.py``` to demonstrate the main functions you have to override to modify the behavior of the agent.

You don't have to modify every method (if you don't and don't want the NotImplementedError to be raised, please remove the function altogether). Instead, simply modify whichever method you need and the others will inherit from ```HumanoidAgent```.

## Future Plans

- [ ] Support customized map and agents on Game Interface
- [ ] Support other LLMs
- [ ] Support other aspects of System 1 thinking

## Citation

```bibtex
@inproceedings{wang-etal-2023-humanoid,
    title = "Humanoid Agents: Platform for Simulating Human-like Generative Agents",
    author = "Wang, Zhilin  and
      Chiu, Yu Ying  and
      Chiu, Yu Cheung",
    editor = "Feng, Yansong  and
      Lefever, Els",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-demo.15",
    doi = "10.18653/v1/2023.emnlp-demo.15",
    pages = "167--176",
    abstract = "Just as computational simulations of atoms, molecules and cells have shaped the way we study the sciences, true-to-life simulations of human-like agents can be valuable tools for studying human behavior. We propose Humanoid Agents, a system that guides Generative Agents to behave more like humans by introducing three elements of System 1 processing: Basic needs (e.g. hunger, health and energy), Emotion and Closeness in Relationships. Humanoid Agents are able to use these dynamic elements to adapt their daily activities and conversations with other agents, as supported with empirical experiments. Our system is designed to be extensible to various settings, three of which we demonstrate, as well as to other elements influencing human behavior (e.g. empathy, moral values and cultural background). Our platform also includes a Unity WebGL game interface for visualization and an interactive analytics dashboard to show agent statuses over time. Our platform is available on https://www.humanoidagents.com/ and code is on https://github.com/HumanoidAgents/HumanoidAgents",
}
```

