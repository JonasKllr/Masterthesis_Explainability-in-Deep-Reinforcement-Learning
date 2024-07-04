import multiprocessing as mp
import numpy as np

def target_function(index, n, queue: mp.Queue, casted_states):
    print("process started")
    print(n)
    queue.put((index, n))

if __name__ == "__main__":
    
    states = np.zeros(shape=(100000, 6))
    casted_states = [dict(enumerate(row)) for row in states]

    processes = []
    queue = mp.Queue()

    for index, explainer in enumerate(range(3)):
        n = index
        prosess = mp.Process(target=target_function, args=(index, n, queue, casted_states))
        processes.append(prosess)
        prosess.start()

    updated_explainers = [None] * len(range(3))
    # while not queue.empty():
    for _ in range(3):
        index, updated_explainer = queue.get()
        updated_explainers[index] = updated_explainer
    
    for process in processes:
        process.join()

    print(updated_explainers)