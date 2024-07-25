import numpy as np
import torch

import torch.multiprocessing as multiprocessing

from ixai.explainer.pdp import IncrementalPDP, BatchPDP


def cast_state_buffer_to_array_of_dicts(state_buffer: np.ndarray):
    return [dict(enumerate(row)) for row in state_buffer]


def explain_one_threading(index, explainer: IncrementalPDP, state_dict_array, queue):
    print("thread start")

    torch.set_num_threads(1)
    for state_dict in state_dict_array:
        explainer.explain_one(state_dict)
        print(state_dict)

    print("state processing done")
    queue.put((index, explainer))


def explain_one_threading_batch(index, explainer: BatchPDP, new_states, queueu):
    print("thread start")

    torch.set_num_threads(1)
    explainer.explain_many(new_states)
    print("state processing done")

    queueu.put((index, explainer))
    print("queue.put() done")


if __name__ == "__main__":
    state_buffer = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    state_dict = cast_state_buffer_to_array_of_dicts(state_buffer)
    print("hello")
