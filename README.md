## Event agent


## Contents

- [Event agent](#event-agent)
- [Contents](#contents)
- [Installation](#installation)
- [Get Started](#get-started)




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
cd scripts
 ..\simulation_args\run.bat philosophy_lecture
```

Windows users: avoid using "\\" in command

Prerequisites

1. Edit ```\scripts\llm.py```, fill in your OpenAI api key,
