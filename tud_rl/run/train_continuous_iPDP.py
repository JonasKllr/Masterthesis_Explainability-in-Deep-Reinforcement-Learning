import csv
import pickle
import random
import shutil
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

from ixai.explainer.pdp import IncrementalPDP
from ixai.storage.ordered_reservoir_storage import OrderedReservoirStorage

import tud_rl.agents.continuous as agents
from tud_rl import logger
from tud_rl.agents.base import _Agent
from tud_rl.common.configparser import ConfigFile
from tud_rl.common.logging_func import EpochLogger
from tud_rl.common.logging_plot import plot_from_progress
from tud_rl.wrappers import get_wrapper
from tud_rl.wrappers.action_selection_wrapper import ActionSelectionWrapper


def evaluate_policy(test_env: gym.Env, agent: _Agent, c: ConfigFile):

    # go greedy
    agent.mode = "test"

    rets = []

    for _ in range(c.eval_episodes):

        # LSTM: init history
        if agent.needs_history:
            s_hist = np.zeros((agent.history_length, agent.state_shape))
            a_hist = np.zeros((agent.history_length, agent.num_actions))
            hist_len = 0

        # get initial state
        s = test_env.reset()

        cur_ret = np.zeros((agent.N_agents, 1)) if agent.is_multi else 0.0
        d = False
        eval_epi_steps = 0

        while not d:

            eval_epi_steps += 1

            # select action
            if agent.needs_history:
                a = agent.select_action(s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
            else:
                a = agent.select_action(s)

            # perform step
            if "UAM" in c.Env.name and agent.name == "LSTMRecTD3":
                s2, r, d, _ = test_env.step(agent)
            else:
                s2, r, d, _ = test_env.step(a)

            # LSTM: update history
            if agent.needs_history:
                if hist_len == agent.history_length:
                    s_hist = np.roll(s_hist, shift=-1, axis=0)
                    s_hist[agent.history_length - 1, :] = s

                    a_hist = np.roll(a_hist, shift=-1, axis=0)
                    a_hist[agent.history_length - 1, :] = a
                else:
                    s_hist[hist_len] = s
                    a_hist[hist_len] = a
                    hist_len += 1

            # s becomes s2
            s = s2
            cur_ret += r

            # break option
            if eval_epi_steps == c.Env.max_episode_steps:
                break

        # append return
        rets.append(cur_ret)

    # continue training
    agent.mode = "train"
    return rets


def train(config: ConfigFile, agent_name: str, compute_iPDP: bool, plot_frequency_iPDP: int):
    """Main training loop."""

    # measure computation time
    start_time = time.time()

    # init envs
    env: gym.Env = gym.make(config.Env.name, **config.Env.env_kwargs)
    test_env: gym.Env = gym.make(config.Env.name, **config.Env.env_kwargs)

    # wrappers
    for wrapper in config.Env.wrappers:
        wrapper_kwargs = config.Env.wrapper_kwargs[wrapper]
        env: gym.Env = get_wrapper(name=wrapper, env=env, **wrapper_kwargs)
        test_env: gym.Env = get_wrapper(name=wrapper, env=test_env, **wrapper_kwargs)

    # get state_shape
    if config.Env.state_type == "image":
        raise NotImplementedError("Currently, image input is not available for continuous action spaces.")

    elif config.Env.state_type == "feature":
        config.state_shape = env.observation_space.shape[0]

    # mode and action details
    config.mode = "train"
    config.num_actions = env.action_space.shape[0]

    # seeding
    env.seed(config.seed)
    test_env.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    if agent_name[-1].islower():
        agent_name_red = agent_name[:-2] + "Agent"
    else:
        agent_name_red = agent_name + "Agent"

    # init agent
    agent_ = getattr(agents, agent_name_red)  # get agent class by name
    agent: _Agent = agent_(config, agent_name)  # instantiate agent

    # possibly load replay buffer for continued training
    if hasattr(config, "prior_buffer"):
        if config.prior_buffer is not None:
            with open(config.prior_buffer, "rb") as f:
                agent.replay_buffer = pickle.load(f)

    # initialize logging
    agent.logger = EpochLogger(alg_str    = agent.name,
                               seed       = config.seed,
                               env_str    = config.Env.name,
                               info       = config.Env.info,
                               output_dir = config.output_dir if hasattr(config, "output_dir") else None)

    agent.logger.save_config({"agent_name": agent.name, **config.config_dict})
    agent.print_params(agent.n_params, case=1)

    # save env-file for traceability
    try:
        entry_point = vars(gym.envs.registry[config.Env.name])["entry_point"][12:]
        shutil.copy2(src="tud_rl/envs/_envs/" + entry_point + ".py", dst=agent.logger.output_dir)
    except:
        logger.warning(
            f"Could not find the env file. Make sure that the file name matches the class name. Skipping..."
        )

    # LSTM: init history
    if agent.needs_history:
        s_hist = np.zeros((agent.history_length, agent.state_shape))
        a_hist = np.zeros((agent.history_length, agent.num_actions))
        hist_len = 0

    # get initial state
    state = env.reset()

    # init episode step counter and episode return
    episode_steps = 0
    episode_return = np.zeros((agent.N_agents, 1)) if agent.is_multi else 0.0


    
    # init iPDP objects

    feature_order = np.arange(start=0, stop=np.shape(state)[0])
    feature_order = feature_order.tolist()
    grid_size = 10
    feature_of_interest = 0

    #TODO agent.select_action() return type AND argument type dict!!
    model_function = ActionSelectionWrapper(agent.select_action)


    if compute_iPDP is True:
        storage = OrderedReservoirStorage(
            store_targets=False,
            size=100,
            constant_probability=1.
        )

        incremental_explainer = IncrementalPDP(
            model_function=model_function,
            feature_names=feature_order,
            gridsize=grid_size,
            dynamic_setting=True,
            smoothing_alpha=0.001,
            pdp_feature=feature_of_interest,
            storage=storage,
            storage_size=10,
            output_key='output',
            pdp_history_interval=1000,
            pdp_history_size=10,
            min_max_grid=True   # True: absolute min max values, False: quantiles
        )
    
        params = {
        'legend.fontsize': 'xx-large',
        'figure.figsize': (7, 7),
        'axes.labelsize': 'xx-large',
        'axes.titlesize': 'xx-large',
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large'
        }

        plt.rcParams.update(params)

    # main loop
    for total_steps in range(config.timesteps):

        episode_steps += 1
        print(total_steps)


        #------------- iPDP ----------------------------------------------------

        # convert state to type dict
        state_iPDP = dict(enumerate(state))
        
        incremental_explainer.explain_one(state_iPDP)

        if total_steps != 0 and total_steps % plot_frequency_iPDP == 0:
            fig, axes = incremental_explainer.plot_pdp(
                title=f"iPDP after {total_steps} samples",
                show_pdp_transition=True,
                show_ice_curves=False,
                y_min=-1.0, y_max=1.0,  #TODO make dependable on. Even necessary?
                x_min=0, x_max=1,
                return_plot=True,
                n_decimals=4,
                x_transform=None, y_transform=None,
                batch_pdp=None,
                y_scale=None,
                y_label='Model Output',
                figsize=None,
                mean_centered_pd=False,
                xticks=None,
                xticklabels=None,
                show_legend=True
            )
            plt.savefig(os.path.join("/media/jonas/SSD_new/CMS/Semester_5/Masterarbeit/code/TUD_RL/experiments/change_detection_plots", f"{total_steps}.pdf"))
            # plt.show()
        
        #------------- iPDP ----------------------------------------------------
        

        # select action
        if total_steps < config.act_start_step:
            if agent.is_multi:
                action = np.random.uniform(low=-1.0, high=1.0, size=(agent.N_agents, agent.num_actions))
            else:
                action = np.random.uniform(low=-1.0, high=1.0, size=agent.num_actions)
        else:
            if agent.needs_history:
                action = agent.select_action(s=state, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
            else:
                action = agent.select_action(state)

        # perform step
        if "UAM" in config.Env.name and agent.name == "LSTMRecTD3":
            state_2, reward, done, _ = env.step(agent)
        else:
            state_2, reward, done, _ = env.step(action)

        # Ignore "done" if it comes from hitting the time horizon of the environment
        done = False if episode_steps == config.Env.max_episode_steps else done

        # add episode return
        episode_return += reward

        # memorize
        agent.memorize(state, action, reward, state_2, done)

        # LSTM: update history
        if agent.needs_history:
            if hist_len == agent.history_length:
                s_hist = np.roll(s_hist, shift=-1, axis=0)
                s_hist[agent.history_length - 1, :] = state

                a_hist = np.roll(a_hist, shift=-1, axis=0)
                a_hist[agent.history_length - 1, :] = action
            else:
                s_hist[hist_len] = state
                a_hist[hist_len] = action
                hist_len += 1

        # train
        if (total_steps >= config.upd_start_step) and (total_steps % config.upd_every == 0):
            agent.train()

        # state becomes state_2
        state = state_2

        # end of episode handling
        if done or (episode_steps == config.Env.max_episode_steps):

            # reset noise after episode
            if hasattr(agent, "noise"):
                agent.noise.reset()

            # LSTM: reset history
            if agent.needs_history:
                s_hist = np.zeros((agent.history_length, agent.state_shape))
                a_hist = np.zeros((agent.history_length, agent.num_actions))
                hist_len = 0

            # reset to initial state
            state = env.reset()

            # log episode return
            if agent.is_multi:
                for i in range(agent.N_agents):
                    agent.logger.store(**{f"Epi_Ret_{i}" : episode_return[i].item()})
            else:
                agent.logger.store(Epi_Ret=episode_return)

            # reset episode steps and episode return
            episode_steps = 0
            episode_return = np.zeros((agent.N_agents, 1)) if agent.is_multi else 0.0

        # end of epoch handling
        if (total_steps + 1) % config.epoch_length == 0 and (total_steps + 1) > config.upd_start_step:

            epoch = (total_steps + 1) // config.epoch_length

            # evaluate agent with deterministic policy
            eval_ret = evaluate_policy(test_env=test_env, agent=agent, c=config)

            if agent.is_multi:
                for ret_list in eval_ret:
                    for i in range(agent.N_agents):
                        agent.logger.store(**{f"Eval_ret_{i}" : ret_list[i].item()})
            else:
                for ret in eval_ret:
                    agent.logger.store(Eval_ret=ret)

            # log and dump tabular
            agent.logger.log_tabular("Epoch", epoch)
            agent.logger.log_tabular("Timestep", total_steps)
            agent.logger.log_tabular("Runtime_in_h", (time.time() - start_time) / 3600)

            if agent.is_multi:
                for i in range(agent.N_agents):
                    agent.logger.log_tabular(f"Epi_Ret_{i}", with_min_and_max=True)
                    agent.logger.log_tabular(f"Eval_ret_{i}", with_min_and_max=True)
                    agent.logger.log_tabular(f"Q_val_{i}", average_only=True)
                    agent.logger.log_tabular(f"Critic_loss_{i}", average_only=True)
                    agent.logger.log_tabular(f"Actor_loss_{i}", average_only=True)
            else:
                agent.logger.log_tabular("Epi_Ret", with_min_and_max=True)
                agent.logger.log_tabular("Eval_ret", with_min_and_max=True)
                agent.logger.log_tabular("Q_val", with_min_and_max=True)
                agent.logger.log_tabular("Critic_loss", average_only=True)
                agent.logger.log_tabular("Actor_loss", average_only=True)

            if agent.needs_history:
                agent.logger.log_tabular("Actor_CurFE", with_min_and_max=False)
                agent.logger.log_tabular("Actor_ExtMemory", with_min_and_max=False)
                agent.logger.log_tabular("Critic_CurFE", with_min_and_max=False)
                agent.logger.log_tabular("Critic_ExtMemory", with_min_and_max=False)

            agent.logger.dump_tabular()

            # create evaluation plot based on current 'progress.txt'
            plot_from_progress(dir     = agent.logger.output_dir,
                               alg     = agent.name,
                               env_str = config.Env.name,
                               info    = config.Env.info)
            # save weights
            save_weights(agent, eval_ret)

# def convert_agent_output_to_dict(fn_action_selection):




def save_weights(agent: _Agent, eval_ret) -> None:

    # check whether this was the best evaluation epoch so far
    with open(f"{agent.logger.output_dir}/progress.txt") as f:
        reader = csv.reader(f, delimiter="\t")
        d = list(reader)

    df = pd.DataFrame(d)
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df = df.astype(float)

    # no best-weight-saving for multi-agent problems since the definition of best weights is not straightforward anymore
    if agent.is_multi:
        best_weights = False
    elif len(df["Avg_Eval_ret"]) == 1:
        best_weights = True
    else:
        if np.mean(eval_ret) > max(df["Avg_Eval_ret"][:-1]):
            best_weights = True
        else:
            best_weights = False

    # usual save
    torch.save(agent.actor.state_dict(), f"{agent.logger.output_dir}/{agent.name}_actor_weights.pth")
    torch.save(agent.critic.state_dict(), f"{agent.logger.output_dir}/{agent.name}_critic_weights.pth")

    # best save
    if best_weights:
        torch.save(agent.actor.state_dict(), f"{agent.logger.output_dir}/{agent.name}_actor_best_weights.pth")
        torch.save(agent.critic.state_dict(), f"{agent.logger.output_dir}/{agent.name}_critic_best_weights.pth")

    # stores the replay buffer
    with open(f"{agent.logger.output_dir}/buffer.pickle", "wb") as handle:
        pickle.dump(agent.replay_buffer, handle)
