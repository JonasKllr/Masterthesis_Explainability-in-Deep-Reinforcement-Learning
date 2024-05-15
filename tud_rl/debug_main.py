from argparse import ArgumentParser

import tud_rl.envs
import tud_rl.run.train_continuous as cont
import tud_rl.run.train_discrete as discr
import tud_rl.run.visualize_continuous as vizcont
import tud_rl.run.visualize_discrete as vizdiscr
from tud_rl.agents import is_discrete, validate_agent
from tud_rl.common.configparser import ConfigFile
from tud_rl.configs.continuous_actions import __path__ as cont_path
from tud_rl.configs.discrete_actions import __path__ as discr_path

# arguments
task = 'train' # train / viz
config_file = 'mountaincar.yaml'
# seed = 
agent_name = 'DQN'
# dqn_wights = 
# actor_weights =
# critic_weigths =

if agent_name[-1].islower():
    agent_name = agent_name[-2]

# check if supplied agent name matches any available agents
validate_agent(agent_name)

# get the configuration file path depending on the chosen mode
base_path = discr_path[0] if is_discrete(agent_name) else cont_path[0]
config_path = f"{base_path}/{config_file}"

# parse the config file
config = ConfigFile(config_path)

# handle maximum episode steps
config.max_episode_handler()

if task == "train":
    if is_discrete(agent_name):
        discr.train(config, agent_name)
    else:
        cont.train(config, agent_name)
elif task == "viz":
    if is_discrete(agent_name):
        vizdiscr.test(config, agent_name)
    else:
        vizcont.test(config, agent_name)
