from sklearn.inspection import partial_dependence



class explainer_PDP:
    def __init__(self) -> None:
        pass


def calculate_pdp(new_states, agent, pdp_feature, grid_size):
    return partial_dependence(agent.select_action, new_states, pdp_feature, grid_resolution=grid_size)
