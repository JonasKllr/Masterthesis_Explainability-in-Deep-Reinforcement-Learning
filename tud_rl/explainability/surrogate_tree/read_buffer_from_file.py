import numpy as np
import os


def read_buffer_from_file(directory: str):
    state_dir = os.path.join(directory, "state_buffer")
    action_dir = os.path.join(directory, "action_buffer")

    states = np.load(state_dir)
    actions = np.load(action_dir)
    return states, actions
