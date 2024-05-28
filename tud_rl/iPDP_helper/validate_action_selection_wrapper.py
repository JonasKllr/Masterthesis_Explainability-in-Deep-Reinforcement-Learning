from tud_rl.agents.base import _Agent
from ixai.explainer.pdp import IncrementalPDP

"""not exactly the same since output value is casted to float (float64) in base Wrapper class and
    agent.select_action() outputs np.float32"""


def vaildate_action_selection_wrapper(
    agent: _Agent, incremental_explainer: IncrementalPDP, state, state_iPDP
):
    output_agent = agent.select_action(state)
    output_wrapper = incremental_explainer.model_function({**state_iPDP})["output"]

    return output_agent, output_wrapper
