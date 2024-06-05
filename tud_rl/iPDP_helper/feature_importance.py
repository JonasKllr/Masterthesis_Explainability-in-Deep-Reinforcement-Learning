import math
import numpy as np


def calculate_feature_importance(pdp_x: dict, pdp_y: dict) -> float:
    grid_points_x, grid_points_y = pdp_x.values(), pdp_y.values()
    mean_centered_pd = np.mean(list(grid_points_y))
    sum_individual_distance = 0

    for n, (grid_point_x, grid_point_y) in enumerate(zip(grid_points_x, grid_points_y)):
        mean_point_pd = grid_point_y - mean_centered_pd
        sum_individual_distance += mean_point_pd**2

    return math.sqrt(1 / (n - 1) * sum_individual_distance)
