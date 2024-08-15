import numpy as np
import os

def save_buffer_to_file(buffer: np.ndarray, save_dir: str, kind: str):
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

