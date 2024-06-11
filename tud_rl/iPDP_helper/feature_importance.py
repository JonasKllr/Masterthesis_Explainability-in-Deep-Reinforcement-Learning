import math
import matplotlib.pyplot as plt
import numpy as np
import os


def calculate_feature_importance(pdp_x: dict, pdp_y: dict) -> np.float32:
    grid_points_x, grid_points_y = pdp_x.values(), pdp_y.values()
    mean_centered_pd = np.mean(list(grid_points_y))
    sum_individual_distance = 0

    for n, (grid_point_x, grid_point_y) in enumerate(zip(grid_points_x, grid_points_y)):
        mean_point_pd = grid_point_y - mean_centered_pd
        sum_individual_distance += mean_point_pd**2

    # standard deviation for samples casted to float32
    return np.float32(math.sqrt(1 / (n - 1) * sum_individual_distance))


def plot_feature_importance(feature_order, feature_importance_array):  # return type??
    plt.barh(
        [f"feature_{i}" for i in feature_order],
        feature_importance_array,
        color="skyblue",
    )
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("PDP-based Feature Importance")
    plt.tight_layout()


def save_feature_importance_to_csv(
    feature_order: list, feature_importance_array: list, total_steps: int, save_dir
) -> None:
    file_dir = os.path.join(save_dir, "feature_importance_pdp.csv")
    array_to_save = np.append(total_steps, feature_importance_array)

    try:
        with open(file_dir, "x") as file:
            headers = ["time_step"] + [f"feature_{i}" for i in feature_order]
            np.savetxt(file, [headers], delimiter=", ", fmt="%s")
    except FileExistsError:
        pass

    with open(file_dir, "a") as file:
        np.savetxt(file, array_to_save.reshape(1, -1), delimiter=",", fmt="%s")


if __name__ == "__main__":
    FEATURE_ORDER = [0, 1, 2, 3, 4, 5]
    FEATURE_IMPORTANCE_ARRAY = [0.5, 0.5, 0.1, 0.5, 0.6, 0.6]
    FEATURE_IMPORTANCE_ARRAY_1 = [0.5, 0.5, 0.1, 0.5, 0.6, 0.6]
    TOTAL_STEPS = 5000
    TOTAL_STEPS_1 = 10000
    SAVE_DIR = (
        "/media/jonas/SSD_new/CMS/Semester_5/Masterarbeit/code/TUD_RL/experiments/iPDP/"
    )

    save_feature_importance_to_csv(
        FEATURE_ORDER, FEATURE_IMPORTANCE_ARRAY, TOTAL_STEPS, SAVE_DIR
    )
    save_feature_importance_to_csv(
        FEATURE_ORDER, FEATURE_IMPORTANCE_ARRAY_1, TOTAL_STEPS_1, SAVE_DIR
    )
