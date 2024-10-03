import csv
import datetime
import os
import pickle
import random
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import gym

import tud_rl.agents.continuous as agents
from tud_rl import logger
from tud_rl.agents.base import _Agent
from tud_rl.common.configparser import ConfigFile
from tud_rl.common.logging_func import EpochLogger
from tud_rl.common.logging_plot import plot_from_progress
from tud_rl.wrappers import get_wrapper
from tud_rl.explainability.action_selection_wrapper import ActionSelectionWrapperAlibi


import shap
from alibi.explainers import ALE, plot_ale, PartialDependence, plot_pd, KernelShap

from tud_rl.explainability.feature_importance import (
    calculate_feature_importance,
    save_feature_importance_to_csv,
    sort_feature_importance_SHAP,
)
from tud_rl.explainability.get_new_states import get_new_states_in_buffer
from tud_rl.explainability.timer import Timer
from tud_rl.explainability.save_buffer import (
    save_buffer_to_file,
    get_actions_on_new_states,
)


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
                a = agent.select_action(
                    s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len
                )
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


def train(config: ConfigFile, agent_name: str):
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
        raise NotImplementedError(
            "Currently, image input is not available for continuous action spaces."
        )

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
    agent.logger = EpochLogger(
        alg_str=agent.name,
        seed=config.seed,
        env_str=config.Env.name,
        info=config.Env.info,
        output_dir=config.output_dir if hasattr(config, "output_dir") else None,
    )

    agent.logger.save_config({"agent_name": agent.name, **config.config_dict})
    agent.print_params(agent.n_params, case=1)

    # save env-file for traceability
    try:
        entry_point = vars(gym.envs.registry[config.Env.name])["entry_point"][12:]
        shutil.copy2(
            src="tud_rl/envs/_envs/" + entry_point + ".py", dst=agent.logger.output_dir
        )
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
    episode_steps = 20
    episode_return = np.zeros((agent.N_agents, 1)) if agent.is_multi else 0.0

    # --------------------------------------------------------------------------------
    # ----------------------------- init explanations --------------------------------
    # --------------------------------------------------------------------------------

    ON_HPC = False

    PDP_CALCULATE = True
    ALE_CALCULATE = True
    SHAP_CALCULATE = True
    SAVE_BUFFER = True

    EXPLAIN_FREQUENCY = 5000
    if ON_HPC:
        EXPLAIN_FREQUENCY = 100000

    feature_order = np.arange(start=0, stop=np.shape(state)[0])
    feature_order = feature_order.tolist()

    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M")

    if ON_HPC:
        PLOT_DIR_PDP = os.path.join("./", now, "pdp/")
        PLOT_DIR_ALE = os.path.join("./", now, "ale/")
        PLOT_DIR_SHAP = os.path.join("./", now, "SHAP/")
        PLOT_DIR_BUFFER = os.path.join("./", now, "buffer/")

        if PDP_CALCULATE:
            if not os.path.exists(PLOT_DIR_PDP):
                os.makedirs(PLOT_DIR_PDP)
        if ALE_CALCULATE:
            if not os.path.exists(PLOT_DIR_ALE):
                os.makedirs(PLOT_DIR_ALE)
        if SHAP_CALCULATE:
            for i in feature_order:
                if not os.path.exists(os.path.join(PLOT_DIR_SHAP, f"feature_{i}")):
                    os.makedirs(os.path.join(PLOT_DIR_SHAP, f"feature_{i}"))
        if SAVE_BUFFER:
            if not os.path.exists(PLOT_DIR_BUFFER):
                os.makedirs(PLOT_DIR_BUFFER)

    else:
        PLOT_DIR_PDP = os.path.join(
            "/media/jonas/SSD_new/CMS/Semester_5/Masterarbeit/code/TUD_RL/experiments/feature_importance",
            now,
            "pdp/",
        )
        PLOT_DIR_ALE = os.path.join(
            "/media/jonas/SSD_new/CMS/Semester_5/Masterarbeit/code/TUD_RL/experiments/feature_importance",
            now,
            "ale/",
        )
        PLOT_DIR_SHAP = os.path.join(
            "/media/jonas/SSD_new/CMS/Semester_5/Masterarbeit/code/TUD_RL/experiments/feature_importance",
            now,
            "SHAP/",
        )
        PLOT_DIR_BUFFER = os.path.join(
            "/media/jonas/SSD_new/CMS/Semester_5/Masterarbeit/code/TUD_RL/experiments/feature_importance",
            now,
            "buffer/",
        )

        if PDP_CALCULATE:
            if not os.path.exists(PLOT_DIR_PDP):
                os.makedirs(PLOT_DIR_PDP)
        if ALE_CALCULATE:
            if not os.path.exists(PLOT_DIR_ALE):
                os.makedirs(PLOT_DIR_ALE)
        if SHAP_CALCULATE:
            for i in feature_order:
                if not os.path.exists(os.path.join(PLOT_DIR_SHAP, f"feature_{i}")):
                    os.makedirs(os.path.join(PLOT_DIR_SHAP, f"feature_{i}"))
        if SAVE_BUFFER:
            if not os.path.exists(PLOT_DIR_BUFFER):
                os.makedirs(PLOT_DIR_BUFFER)

    agent.mode = "test"
    model_function = ActionSelectionWrapperAlibi(
        action_selection_function=agent.select_action
    )
    agent.mode = "train"

    feature_names = []
    for i in feature_order:
        feature_name = f"feature_{i}"
        feature_names.append(feature_name)

    if PDP_CALCULATE:
        pdp_timer = Timer()
    if ALE_CALCULATE:
        ale_timer = Timer()
    if SHAP_CALCULATE:
        SHAP_timer = Timer()

    # --------------------------------------------------------------------------------
    # ----------------------------- init explanations end ----------------------------
    # --------------------------------------------------------------------------------

    # main loop
    for total_steps in range(config.timesteps):

        episode_steps += 1

        # --------------------------------------------------------------------------------
        # ------------- explanations -----------------------------------------------------
        # --------------------------------------------------------------------------------
        if not ON_HPC:
            print(total_steps)

        # calculate explanations with a frequency of EXPLAIN_FREQUENCY
        if total_steps != 0 and total_steps % EXPLAIN_FREQUENCY == 0:
            agent.mode = "test"
            new_states = get_new_states_in_buffer(
                agent.replay_buffer.s, agent.replay_buffer.ptr, EXPLAIN_FREQUENCY
            )

            if PDP_CALCULATE:
                print("calculating PDP")
                pdp_timer.start_time = time.time()

                pdp_explainer = PartialDependence(
                    predictor=model_function,
                    feature_names=feature_names,
                    target_names=["Partial Dependence"],
                )

                GRID_SIZE = 5
                pdp_explanations = pdp_explainer.explain(
                    X=new_states,
                    features=None,
                    kind="average",
                    grid_resolution=GRID_SIZE,
                )
                plot_pd(pdp_explanations, pd_limits=[-1.0, 1.0])
                plt.savefig(os.path.join(PLOT_DIR_PDP, f"{total_steps}.pdf"))
                plt.clf()
                plt.close("all")

                feature_importance_pdp = [None] * len(feature_order)
                for i in feature_order:
                    feature_importance_pdp[i] = calculate_feature_importance(
                        y_values=pdp_explanations.pd_values[i][0, :],
                    )
                save_feature_importance_to_csv(
                    feature_order,
                    feature_importance_pdp,
                    total_steps,
                    PLOT_DIR_PDP,
                )

                pdp_timer.end_time = time.time()
                pdp_timer.calculate_elapsed_time()
                pdp_timer.update_time_accumulated()
                pdp_timer.save_time_to_csv(total_steps, PLOT_DIR_PDP)

            if ALE_CALCULATE:
                print("calculating ALE")
                ale_timer.start_time = time.time()

                ale_explainer = ALE(
                    predictor=model_function,
                    feature_names=feature_names,
                    target_names=["ALE"],
                )
                ale_explanations = ale_explainer.explain(
                    X=new_states, min_bin_points=10
                )
                axes = plot_ale(ale_explanations, n_cols=3)

                # remove legend from ALE plots
                for ax in axes.ravel():
                    legend = ax.get_legend()
                    if legend:
                        legend.remove()

                # adapt plots to range of ALE values
                min_ale_value = np.min(np.concatenate(ale_explanations.ale_values))
                max_ale_value = np.max(np.concatenate(ale_explanations.ale_values))
                if (min_ale_value < -1) or (max_ale_value > 1):
                    plt.ylim(min_ale_value, max_ale_value)
                else:
                    plt.ylim(-1, 1)
                plt.savefig(os.path.join(PLOT_DIR_ALE, f"{total_steps}.pdf"))
                plt.close("all")

                feature_importance_ale = [None] * len(feature_order)
                for i in feature_order:
                    feature_importance_ale[i] = calculate_feature_importance(
                        np.reshape(ale_explanations.ale_values[i], (-1,))
                    )
                save_feature_importance_to_csv(
                    feature_order,
                    feature_importance_ale,
                    total_steps,
                    PLOT_DIR_ALE,
                )

                ale_timer.end_time = time.time()
                ale_timer.calculate_elapsed_time()
                ale_timer.update_time_accumulated()
                ale_timer.save_time_to_csv(total_steps, PLOT_DIR_ALE)

            if SHAP_CALCULATE:
                print("calculating SHAP")
                SHAP_timer.start_time = time.time()

                shap_explainer = KernelShap(
                    predictor=model_function,
                    link="identity",
                    feature_names=feature_names,
                    task="regression",
                )
                shap_explainer.fit(
                    background_data=new_states,
                    summarise_background=True,
                    n_background_samples=200,
                )

                # getting 200 randomly sampled data points
                random_sample_id = np.random.choice(
                    new_states.shape[0], size=200, replace=False
                )
                shap_explanations = shap_explainer.explain(
                    X=new_states[random_sample_id]
                )

                for i in feature_order:
                    shap.dependence_plot(
                        ind=i,
                        shap_values=shap_explanations.shap_values[0],
                        features=new_states[random_sample_id],
                        feature_names=feature_names,
                        interaction_index=None,
                        show=False,
                    )
                    plt.savefig(
                        os.path.join(
                            PLOT_DIR_SHAP, f"feature_{i}", f"{total_steps}.pdf"
                        ),
                        bbox_inches="tight",
                    )
                    plt.clf()
                plt.close("all")

                ranked_feature_importance = shap_explanations.raw["importances"]["0"]
                sorted_feature_importance = sort_feature_importance_SHAP(
                    ranked_feature_importance
                )
                save_feature_importance_to_csv(
                    feature_order, sorted_feature_importance, total_steps, PLOT_DIR_SHAP
                )

                SHAP_timer.end_time = time.time()
                SHAP_timer.calculate_elapsed_time()
                SHAP_timer.update_time_accumulated
                SHAP_timer.save_time_to_csv(total_steps, PLOT_DIR_SHAP)

            if SAVE_BUFFER:
                print("saving buffer to file")
                new_actions = get_actions_on_new_states(new_states, agent)

                save_buffer_to_file(new_states, PLOT_DIR_BUFFER, kind="states")
                save_buffer_to_file(new_actions, PLOT_DIR_BUFFER, kind="actions")

            agent.mode = "train"
        # --------------------------------------------------------------------------------
        # ------------- explanations end -------------------------------------------------
        # --------------------------------------------------------------------------------

        # select action
        if total_steps < config.act_start_step:
            if agent.is_multi:
                action = np.random.uniform(
                    low=-1.0, high=1.0, size=(agent.N_agents, agent.num_actions)
                )
            else:
                action = np.random.uniform(low=-1.0, high=1.0, size=agent.num_actions)
        else:
            if agent.needs_history:
                action = agent.select_action(
                    s=state, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len
                )
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
        if (total_steps >= config.upd_start_step) and (
            total_steps % config.upd_every == 0
        ):
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
                    agent.logger.store(**{f"Epi_Ret_{i}": episode_return[i].item()})
            else:
                agent.logger.store(Epi_Ret=episode_return)

            # reset episode steps and episode return
            episode_steps = 0
            episode_return = np.zeros((agent.N_agents, 1)) if agent.is_multi else 0.0

        # end of epoch handling
        if (total_steps + 1) % config.epoch_length == 0 and (
            total_steps + 1
        ) > config.upd_start_step:

            epoch = (total_steps + 1) // config.epoch_length

            # evaluate agent with deterministic policy
            eval_ret = evaluate_policy(test_env=test_env, agent=agent, c=config)

            if agent.is_multi:
                for ret_list in eval_ret:
                    for i in range(agent.N_agents):
                        agent.logger.store(**{f"Eval_ret_{i}": ret_list[i].item()})
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
            plot_from_progress(
                dir=agent.logger.output_dir,
                alg=agent.name,
                env_str=config.Env.name,
                info=config.Env.info,
            )
            # save weights
            save_weights(agent, eval_ret)


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
    torch.save(
        agent.actor.state_dict(),
        f"{agent.logger.output_dir}/{agent.name}_actor_weights.pth",
    )
    torch.save(
        agent.critic.state_dict(),
        f"{agent.logger.output_dir}/{agent.name}_critic_weights.pth",
    )

    # best save
    if best_weights:
        torch.save(
            agent.actor.state_dict(),
            f"{agent.logger.output_dir}/{agent.name}_actor_best_weights.pth",
        )
        torch.save(
            agent.critic.state_dict(),
            f"{agent.logger.output_dir}/{agent.name}_critic_best_weights.pth",
        )

    # stores the replay buffer
    with open(f"{agent.logger.output_dir}/buffer.pickle", "wb") as handle:
        pickle.dump(agent.replay_buffer, handle)
