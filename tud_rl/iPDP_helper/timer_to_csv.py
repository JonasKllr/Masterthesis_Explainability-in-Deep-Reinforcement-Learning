import numpy as np
import os

def save_timer_to_csv(timer: float, total_steps: int, save_dir: str):
    file_dir = os.path.join(save_dir, "explanation_timer.csv")
    array_to_save = np.append(total_steps, timer)
    try:
        with open(file_dir, "x") as file:
            headers = ["time_step"] + ["elapsed_time_accumulated"]
            np.savetxt(file, [headers], delimiter=", ", fmt="%s")
    except FileExistsError:
        pass

    with open(file_dir, "a") as file:
        np.savetxt(file, array_to_save.reshape(1, -1), delimiter=",", fmt="%s")


