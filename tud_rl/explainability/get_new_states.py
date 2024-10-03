import numpy as np


def get_new_states_in_buffer(
    state_buffer: np.ndarray, state_ptr: int, plot_frequency_iPDP: int
):
    # state_ptr points to element that will be overriden in next iteration
    new_states_id = np.arange(
        start=((state_ptr - 1) - plot_frequency_iPDP), stop=(state_ptr - 1)
    )
    return state_buffer.take(indices=new_states_id, axis=0, mode="wrap")
