# Explainability in Deep Reinforcement Learning

### Abstract
With the combination of Reinforcement Learning and Artificial Neural Networks, Deep Reinforcement Learning (DRL) agents are shifted towards being non-interpretable black-box models.
Developers of DRL agents, however, could benefit from enhanced interpretability of the agents' behavior, especially during the training process.
Improved interpretability  could enable developers to make informed adaptations, leading to better overall performance.
The explainability methods Partial Dependence Plot (PDP), Accumulated Local Effects (ALE) and SHapley Additive ex-
Planations (SHAP) were considered to provide insights into how an agent's  behavior evolves during training.
Additionally, a decision tree as a surrogate model was considered to enhance the interpretability of a trained agent.
In a case study, the methods were tested on a Deep Deterministic Policy Gradient (DDPG) agent that was trained in an obstacle avoidance scenario.
PDP, ALE and SHAP were evaluated towards their ability to provide explanations as well as the feasibility of their application in terms of computational overhead.
The decision tree was evaluated towards its ability to approximate the agent's policy as a post-hoc method.
Results demonstrated that PDP, ALE and SHAP were able to provide valuable explanations during the training.
Each method contributed additional information with their individual advantages.
However, the decision tree failed to approximate the agent's actions effectively to be used as a surrogate model.