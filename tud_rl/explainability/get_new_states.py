import numpy as np


def get_new_states_from_buffer(
    state_buffer: np.ndarray, state_ptr: int, explain_frequency: int
):
    # state_ptr points to element that will be overriden in next iteration
    new_states_id = np.arange(
        start=((state_ptr - 1) - explain_frequency), stop=(state_ptr - 1)
    )
    return state_buffer.take(indices=new_states_id, axis=0, mode="wrap")
