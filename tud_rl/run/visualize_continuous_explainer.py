import random
import pickle
import os

import gym
import numpy as np
import torch

import tud_rl.agents.continuous as agents
from tud_rl.agents.base import _Agent
from tud_rl.common.configparser import ConfigFile
from tud_rl.wrappers import get_wrapper


def visualize_policy(env: gym.Env, env_tree: gym.Env, agent: _Agent, c: ConfigFile):
    BUFFER_DIR = "/media/jonas/SSD_new/CMS/Semester_5/Masterarbeit/plots/final/explanations_4mil_final/plots/explainer_1_interrupted/2024-08-15_22-00/buffer"
    with open(os.path.join(BUFFER_DIR, 'surrogate_tree_depth-20.pkl'), 'rb') as file:
    # with open(os.path.join(BUFFER_DIR, 'trees', 'depth_4', 'surrogate_tree_model.pkl'), 'rb') as file:
        surrogate_tree = pickle.load(file)

    actions_agent = []
    actions_tree = []
    states = []

    for i in range(c.eval_episodes):

        # LSTM: init history
        if agent.needs_history:
            s_hist = np.zeros((agent.history_length, agent.state_shape))
            a_hist = np.zeros((agent.history_length, agent.num_actions))
            hist_len = 0

        # get initial state
        s = env.reset()
        s_tree = env_tree.reset()

        cur_ret = 0
        cur_ret_tree = 0

        d = False
        d_tree = False
        eval_epi_steps = 0

        while not d:

            eval_epi_steps += 1

            # render env
            # env_tree.render()
            env.render()

            # select action
            a_tree = surrogate_tree.predict(np.reshape(s, (1, -1)))
            a = agent.select_action(s)

            if i == 9:
                actions_agent.append(a)
                actions_tree.append(a_tree)
                states.append(s)



            # perform step
            # s2_tree, r_tree, d_tree, _ =env_tree.step(a_tree)
            s2, r, d, _ = env.step(a)

            # s becomes s2
            # s_tree = s2_tree
            # cur_ret_tree += r_tree
            s = s2
            cur_ret = r

            # break option
            if eval_epi_steps == c.Env.max_episode_steps:
                break
        print(cur_ret)
        # print(f"tree return: {cur_ret_tree}")
    
    actions_agent = np.array(actions_agent)
    actions_tree = np.array(actions_tree)
    states = np.array(states)

    with open(os.path.join(BUFFER_DIR, "comparision_agetn_tree", 'actions_agent.csv'), 'xb') as file:
        np.savetxt(file, actions_agent, fmt='%1.5f')
    
    with open(os.path.join(BUFFER_DIR, "comparision_agetn_tree", 'actions_tree.csv'), 'xb') as file:
        np.savetxt(file, actions_tree, fmt='%1.5f')
    
    with open(os.path.join(BUFFER_DIR, "comparision_agetn_tree", 'states.csv'), 'xb') as file:
        np.savetxt(file, states, fmt='%1.5f')


def test(c: ConfigFile, agent_name: str):
    # init envs
    env = gym.make(c.Env.name, **c.Env.env_kwargs)
    env_tree = gym.make(c.Env.name, **c.Env.env_kwargs)

    # wrappers
    for wrapper in c.Env.wrappers:
        wrapper_kwargs = c.Env.wrapper_kwargs[wrapper]
        env: gym.Env = get_wrapper(name=wrapper, env=env, **wrapper_kwargs)


    # get state shape
    if c.Env.state_type == "image":
        raise NotImplementedError("Currently, image input is not available for continuous action spaces.")

    elif c.Env.state_type == "feature":
        c.state_shape = env.observation_space.shape[0]

    # mode and action details
    c.mode = "test"
    c.num_actions = env.action_space.shape[0]

    # seeding
    env.seed(c.seed)
    env_tree.seed(c.seed)
    torch.manual_seed(c.seed)
    np.random.seed(c.seed)
    random.seed(c.seed)

    # agent prep
    if agent_name[-1].islower():
        agent_name_red = agent_name[:-2] + "Agent"
    else:
        agent_name_red = agent_name + "Agent"

    # init agent
    agent_ = getattr(agents, agent_name_red)  # Get agent class by name
    agent: _Agent = agent_(c, agent_name)  # Instantiate agent

    # visualization
    visualize_policy(env=env, env_tree=env_tree, agent=agent, c=c)
