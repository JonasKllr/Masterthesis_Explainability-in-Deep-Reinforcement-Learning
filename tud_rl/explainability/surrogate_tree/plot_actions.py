import os 
import numpy as np
import matplotlib.pyplot as plt

ACTIONS_DIR = "/media/jonas/SSD_new/CMS/Semester_5/Masterarbeit/plots/final/explanations_4mil_final/plots/explainer_1_interrupted/2024-08-15_22-00/buffer/comparision_agetn_tree/depth_6/good_nr-9"

actions_agent = np.loadtxt(os.path.join(ACTIONS_DIR, "actions_agent.csv"))
actions_tree = np.loadtxt(os.path.join(ACTIONS_DIR, "actions_tree.csv"))

timesteps = np.arange(np.shape(actions_agent)[0])

plt.plot(timesteps, actions_agent, linewidth=3.0, color="#8BC1F7", label="agent")
plt.plot(timesteps, actions_tree, linewidth=3.0, color="#4CB140", label="surrogate tree")

plt.xlabel("time step [-]")
plt.ylabel("action [-]")
plt.legend()
plt.show()