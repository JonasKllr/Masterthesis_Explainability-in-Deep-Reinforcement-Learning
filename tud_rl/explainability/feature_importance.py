import math
import matplotlib.pyplot as plt
import numpy as np
import os


def calculate_feature_importance(y_values: np.ndarray) -> np.float32:
    mean_centered_pd = np.mean(y_values)
    mean_point_pd = y_values - mean_centered_pd
    sum_individual_distance = np.sum(np.power(mean_point_pd, 2))
    number_grid_points = np.shape(y_values)[0]

    # standard deviation for samples casted to float32
    return np.float32(math.sqrt(1 / (number_grid_points - 1) * sum_individual_distance))


def calculate_feature_importance_iPDP(pdp_x: dict, pdp_y: dict) -> np.float32:
    grid_points_x, grid_points_y = pdp_x.values(), pdp_y.values()
    mean_centered_pd = np.mean(list(grid_points_y))
    sum_individual_distance = 0
    for n, (grid_point_x, grid_point_y) in enumerate(zip(grid_points_x, grid_points_y)):
        mean_point_pd = grid_point_y - mean_centered_pd
        sum_individual_distance += mean_point_pd**2

    # standard deviation for samples casted to float32
    return np.float32(math.sqrt(1 / (n - 1) * sum_individual_distance))


def plot_feature_importance(feature_order, feature_importance_array) -> None:
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
    feature_order: list, feature_importance: list, total_steps: int, save_dir: str
) -> None:
    file_dir = os.path.join(save_dir, "feature_importance_.csv")
    array_to_save = np.append(total_steps, feature_importance)
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
