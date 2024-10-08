# Explainability in Deep Reinforcement Learning

This folder contains the helper scripts that were applied in the project.
The applications of the methods during the agent's training are in [train_continuos_explainer.py](../run/train_continuous_explainer.py) and [train_continuos_iPDP.py](../run/train_continuous_iPDP.py).

## Abstract
With the combination of Reinforcement Learning (RL) and Artificial Neural Networks, Deep Reinforcement Learning (DRL) agents are shifted towards being non-interpretable black-box models.
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

## Case Study: Obstacle Avoidance
The RL agent was trained in a simple two-dimensional environment consisting of two moving obstacles with a constant distance relative to each other.
The agent's aim was to pass in between the obstacles. [1]

|<img src="./img/env.png" alt="drawing" width="300"/>|
|:--:|
|Obstacle avoidance environment taken from [1].|

The state was represented by six features:
$$s_{t} =\left(\begin{array}{c}
		\frac{\ddot{y}_{t,agent}}{a_{y,max}} \\
		\frac{\dot{y}_{t,agent}}{v_{y,max}} \\
		\frac{\dot{x}_{t,agent}-\dot{x_{t,i}}}{v_{x,max}} \\
		\frac{\dot{y}_{t,agent}-\dot{y_{t,i}}}{v_{y,max}} \\
		\frac{x_{t,agent}-x_{t,i}}{x_{scale}} \\
		\frac{y_{t,agent}-y_{t,i}}{y_{scale}}
	\end{array}
	\right)$$

Based on the state representation, the agent computed an action $a_{t} \in [-1,1]$.
This action was then mapped to the agent's acceleration in lateral direction:
$$\ddot{y}_{t+1,agent} = \ddot{y}_{t,agent} + \Delta a_{y,max} \cdot a_{t}$$


The DRL agent was trained for $4 \cdot 10^{6}$ time steps.
During the training, the above mentioned explainability methods PDP, ALE and SHAP where applied with a frequency of $10^{5}$ time steps.
At each evaluation step, the states encountered by the agent in the the previous $10^{5}$ time steps were used as the data set for the calcualtion of the explainability methods.
Furthermore, the incremental PDP (iPDP) [2], an adaption of the PDP to dynamic modeling scearios, was applied at every time step.
The aim was to investigate, how the importance of the individual feautres for the agent's decisions changed over time with a progressing training process.

A decision tree was applied post-hoc as a surrogate model to the agent to investigate, if this method can provide further explanations about the agent's policy.

## Results

### Methods applied during the training
The iPDP resulted in an average increase of runtime of 10.32 hours compared to training without the applicaiton of any explainability methods.
This was considered to be unfeasable for the method to work in this scenario.
Thereofore, no further investigations were conducted on the iPDP.

PDP, ALE and SHAP produced similar results in terms of feature importance of the individual feautres:

|PDP Results|
|:--:|
|<img src="./img/pdp.png" alt="drawing" width="550"/>|
|PDPs of the individual features in the state representation at time step $2 \cdot 10^{6}$.|
|<img src="./img/pdp_feature-importance.png" alt="drawing" width="650"/>|
|Feature importance calculated based on the PDP over the agent's training process for $4 \cdot 10^{6}$ time steps.|

|ALE Results|
|:--:|
|<img src="./img/ale.png" alt="drawing" width="550"/>|
|ALEs of the individual features in the state representation at time step $2 \cdot 10^{6}$.|
|<img src="./img/ale_feature-importance.png" alt="drawing" width="650"/>|
|Feature importance calculated based on the ALE over the agent's training process for $4 \cdot 10^{6}$ time steps.|

|SHAP Results|
|:--:|
|<img src="./img/ale.png" alt="drawing" width="550"/>|
|SHAP dependence plots of the individual features in the state representation at time step $2 \cdot 10^{6}$. 200 data points were used for the calculation.|
|<img src="./img/ale_feature-importance.png" alt="drawing" width="650"/>|
|Feature importance calculated based on SHAP over the agent's training process for $4 \cdot 10^{6}$ time steps.|


<!-- true computational costs -->
The application of the explainability add computational costs to the agent's training process.
SHAP introduces by far the highest computational costs, followed by PDP.
Compared to the other methods, ALE introduces the lowest computational costs.

<!-- computational costs here -->
The methods were applied in a way to address for the trade-off between reliable results by including a high number of data points and keeping the computational costs low.
At each evaluation step during the training, PDP and ALE were both calculated with the same set of $10^{5}$ data points.
The PDP was calculated for five grid points, whereas the number of grid points for the ALE calculation was determined algorithmically such that ten data points fall in each interval vetween two adjacent grid points.
SHAP was calculated on 200 data points radomly sampled from the $10^{5}$ data points used for PDP and ALE.

|<img src="./img/runtimes.png" alt="drawing" width="500"/>|
|:--:|
|Runtimes of the methods PDP, ALE and SHAP during a single evaluation step at every $10^{5}$ time steps. The runtimes were measured during three training runs with the same settings.|

## Surrogate Model
## Conclusion

### Bibliography

[1] Hart, Fabian, Martin Waltz and Ostap Okhrin: Missing Velocity in Dynamic Obstacle Avoidance based on Deep Reinforcement Learning. arXiv preprint arXiv:2112.12465, 2021.

[2] Muschalik, Maximilian, Fabian Fumagalli, Rohit Jagtani, Barbara
Hammer and Eyke Hüllermeier: iPDP: On Partial Dependence Plots in Dy-
namic Modeling Scenarios. In World Conference on Explainable Artificial Intelli-
gence, pages 177–194. Springer, 2023.