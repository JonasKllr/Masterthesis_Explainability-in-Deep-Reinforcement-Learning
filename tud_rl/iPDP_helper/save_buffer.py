import numpy as np
import os

from tud_rl.agents.base import _Agent


def get_actions_on_new_states(new_states: np.ndarray, agent: _Agent) -> np.ndarray:
    new_actions = np.zeros(shape=np.shape(new_states)[0])
    for i in range(np.shape(new_states)[0]):
        new_actions[i] = agent.select_action(new_states[i, :])

    return new_actions


def save_buffer_to_file(buffer: np.ndarray, save_dir: str, kind: str) -> None:
    if kind == "states":
        file_dir = os.path.join(save_dir, "state_buffer")
    elif kind == "actions":
        file_dir = os.path.join(save_dir, "action_buffer")

    try:
        with open(file_dir, "xb") as file:
            np.save(file=file, arr=buffer)
    except FileExistsError:
        pass

    with open(file_dir, "wb") as file:
        np.save(file=file, arr=buffer)
