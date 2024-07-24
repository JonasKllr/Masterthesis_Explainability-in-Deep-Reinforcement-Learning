import math
import matplotlib.pyplot as plt
import numpy as np
import os


def calculate_feature_importance(y_values: np.ndarray) -> np.float32:
    mean_centered_pd = np.mean(y_values)
    sum_individual_distance = 0

    mean_point_pd = y_values - mean_centered_pd
    sum_individual_distance = np.sum(np.power(mean_point_pd, 2))

    # for n, (x_values, y_values) in enumerate(zip(x_values, y_values)):
    #     mean_point_pd = y_values - mean_centered_pd
    #     sum_individual_distance += mean_point_pd**2

    # standard deviation for samples casted to float32
    number_grid_points = np.shape(y_values)[0]
    return np.float32(math.sqrt(1 / (number_grid_points - 1) * sum_individual_distance))


def calculate_feature_importance_ale(feature_values: np.ndarray, ale_vales: np.ndarray):
    mean_centered_ale = np.mean(ale_vales)
    mean_point_ale = ale_vales - mean_centered_ale
    sum_individual_distance = np.sum(np.power(mean_point_ale, 2), dtype=np.float32)

    n = feature_values.shape[0]
    return math.sqrt(1 / (n - 1) * sum_individual_distance)


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


def save_feature_importance_to_csv_pdp(
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


def save_feature_importance_to_csv_ale(
    feature_order: list, feature_importance_array: list, total_steps: int, save_dir
) -> None:
    file_dir = os.path.join(save_dir, "feature_importance_ale.csv")
    array_to_save = np.append(total_steps, feature_importance_array)

    try:
        with open(file_dir, "x") as file:
            headers = ["time_step"] + [f"feature_{i}" for i in feature_order]
            np.savetxt(file, [headers], delimiter=", ", fmt="%s")
    except FileExistsError:
        pass

    with open(file_dir, "a") as file:
        np.savetxt(file, array_to_save.reshape(1, -1), delimiter=",", fmt="%s")


def save_feature_importance_to_csv_SHAP(
    feature_order: list, feature_importance_array: list, total_steps: int, save_dir
) -> None:
    file_dir = os.path.join(save_dir, "feature_importance_SHAP.csv")
    array_to_save = np.append(total_steps, feature_importance_array)

    try:
        with open(file_dir, "x") as file:
            headers = ["time_step"] + [f"feature_{i}" for i in feature_order]
            np.savetxt(file, [headers], delimiter=", ", fmt="%s")
    except FileExistsError:
        pass

    with open(file_dir, "a") as file:
        np.savetxt(file, array_to_save.reshape(1, -1), delimiter=",", fmt="%s")


def sort_feature_importance_SHAP(feature_importance_SHAP):
    ranked_effects = feature_importance_SHAP["ranked_effect"]
    names = feature_importance_SHAP["names"]

    combined = list(zip(names, ranked_effects))
    combined_sorted = sorted(combined, key=lambda x: x[0])
    _, sorted_effects = zip(*combined_sorted)

    return sorted_effects

def save_feature_importance_to_csv_tree(
    feature_order: list, feature_importance_array: list, total_steps: int, save_dir
) -> None:
    file_dir = os.path.join(save_dir, "feature_importance_tree.csv")
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

    # save_feature_importance_to_csv(
    #     FEATURE_ORDER, FEATURE_IMPORTANCE_ARRAY, TOTAL_STEPS, SAVE_DIR
    # )
    # save_feature_importance_to_csv(
    #     FEATURE_ORDER, FEATURE_IMPORTANCE_ARRAY_1, TOTAL_STEPS_1, SAVE_DIR
    # )
