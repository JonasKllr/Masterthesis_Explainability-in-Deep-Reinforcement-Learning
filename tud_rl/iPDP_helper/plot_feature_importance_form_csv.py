import pandas as pd
import matplotlib.pyplot as plt

DIR_CSV = "/media/jonas/SSD_new/CMS/Semester_5/Masterarbeit/plots/2024-06-11_21-05/feature_importance_pdp.csv"
data = pd.read_csv(DIR_CSV)

# exponential moving average
span = 4  # 1 for no smoothing
data = data.ewm(span=span, adjust=False).mean()

plt.figure(figsize=(10, 6))
for column in data.columns[1:]:
    plt.plot(data["time_step"], data[column], label=column)

plt.xlabel("Time Step")
plt.ylabel("Feature Importance")
plt.title(
    r"$\bf{iPDP-based\ Feature\ Importance}$"
    + "\n"
    + rf"exponentially smoothed with $\alpha = \dfrac{{2}}{{{span}+1}}$"
)
plt.legend()
plt.grid(True)
plt.show()
