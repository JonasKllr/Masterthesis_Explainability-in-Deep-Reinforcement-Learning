import os
import numpy as np
import matplotlib.pyplot as plt

ACTIONS_DIR = "path/to/folder"

actions_agent = np.loadtxt(os.path.join(ACTIONS_DIR, "actions_agent.csv"))
actions_tree = np.loadtxt(os.path.join(ACTIONS_DIR, "actions_tree.csv"))

timesteps = np.arange(np.shape(actions_agent)[0])

plt.plot(timesteps, actions_agent, linewidth=3.0, color="#8BC1F7", label="agent")
plt.plot(
    timesteps, actions_tree, linewidth=3.0, color="#4CB140", label="surrogate tree"
)

plt.xlabel("time step [-]")
plt.ylabel("action [-]")
plt.legend()
plt.show()
