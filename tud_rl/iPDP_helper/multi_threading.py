import numpy as np

from multiprocessing import Queue


def cast_state_buffer_to_array_of_dicts(state_buffer: np.ndarray):
    return [dict(enumerate(row)) for row in state_buffer]


def get_new_states_in_buffer(
    state_buffer: np.ndarray, state_ptr: int, plot_frequency_iPDP: int
):
    # state_ptr points to element that will be overriden in next iteration
    new_states_id = np.arange(
        start=((state_ptr - 1) - plot_frequency_iPDP), stop=(state_ptr - 1)
    )
    return state_buffer.take(indices=new_states_id, axis=0, mode="wrap")


def explain_one_threading(index,  explainer, state_dict_array, queue: Queue):
    print("thread start")

    for state_dict in state_dict_array:
        explainer.explain_one(state_dict)
        print(state_dict)

    queue.put((index, explainer))


if __name__ == "__main__":
    state_buffer = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    state_dict = cast_state_buffer_to_array_of_dicts(state_buffer)
    print("hello")
