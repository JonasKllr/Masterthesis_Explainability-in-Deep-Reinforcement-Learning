import numpy as np
import os

from dataclasses import dataclass


@dataclass
class Timer:
    time_accumulated: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    time_elapsed: float = 0.0

    def calculate_elapsed_time(self):
        self.time_elapsed = self.end_time - self.start_time

    def update_time_accumulated(self):
        self.time_accumulated += self.time_elapsed

    def save_time_to_csv(self, total_steps: int, save_dir: str):
        file_dir = os.path.join(save_dir, "explanation_timer.csv")
        array_to_save = np.append(total_steps, self.time_accumulated)
        try:
            with open(file_dir, "x") as file:
                headers = ["time_step"] + ["elapsed_time_accumulated"]
                np.savetxt(file, [headers], delimiter=", ", fmt="%s")
        except FileExistsError:
            pass

        with open(file_dir, "a") as file:
            np.savetxt(file, array_to_save.reshape(1, -1), delimiter=",", fmt="%s")
