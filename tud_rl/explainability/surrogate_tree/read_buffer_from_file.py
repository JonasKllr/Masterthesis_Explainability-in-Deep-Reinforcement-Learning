import numpy as np
import os


def read_buffer_from_file(directory: str):
    state_dir = os.path.join(directory, "state_buffer")
    action_dir = os.path.join(directory, "action_buffer")    

    states = np.load(state_dir)
    actions = np.load(action_dir)
    return states, actions



if __name__ == "__main__":
    path_dir = "/media/jonas/SSD_new/CMS/Semester_5/Masterarbeit/plots/final/explanations_4mil_timeout/plots/explainer_1/2024-08-15_22-00/buffer"
    state_dir = os.path.join(path_dir, "state_buffer")
    action_dir = os.path.join(path_dir, "action_buffer")

    states = np.load(state_dir)
    actions = np.load(action_dir)

    print(states)
    print(actions)