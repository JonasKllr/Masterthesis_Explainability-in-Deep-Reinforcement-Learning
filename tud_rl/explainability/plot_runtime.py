import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)

    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    plt.legend()


DIR_CSV = "/media/jonas/SSD_new/CMS/Semester_5/Masterarbeit/plots/final/explanations_4mil_final/plots/runtimes"

ALE_CSV = os.path.join(DIR_CSV, "ale.csv")
PDP_CSV = os.path.join(DIR_CSV, "pdp.csv")
SHAP_CSV = os.path.join(DIR_CSV, "shap.csv")

HEADER_NAMES = ["Training 1", "Training 2", "Training 3"]

df_ale = pd.read_csv(ALE_CSV, header=0, names=HEADER_NAMES, index_col=False)
df_pdp = pd.read_csv(PDP_CSV, header=0, names=HEADER_NAMES, index_col=False)
df_shap = pd.read_csv(SHAP_CSV, header=0, names=HEADER_NAMES, index_col=False)

df_ale["Training 1"] = df_ale["Training 1"].diff()
df_ale["Training 2"] = df_ale["Training 2"].diff()
df_ale["Training 3"] = df_ale["Training 3"].diff()

df_pdp["Training 1"] = df_pdp["Training 1"].diff()
df_pdp["Training 2"] = df_pdp["Training 2"].diff()
df_pdp["Training 3"] = df_pdp["Training 3"].diff()

df_shap["Training 1"] = df_shap["Training 1"].diff()
df_shap["Training 2"] = df_shap["Training 2"].diff()
df_shap["Training 3"] = df_shap["Training 3"].diff()

df_ale = df_ale.dropna()
df_pdp = df_pdp.dropna()
df_shap = df_shap.dropna()

data_to_plot_1 = [df_ale["Training 1"], df_pdp["Training 1"], df_shap["Training 1"]]
data_to_plot_2 = [df_ale["Training 2"], df_pdp["Training 2"], df_shap["Training 2"]]
data_to_plot_3 = [df_ale["Training 3"], df_pdp["Training 3"], df_shap["Training 3"]]

# grouped box plot
ale_plot = plt.boxplot(
    data_to_plot_1, showfliers=False, positions=[0.5, 2.5, 4.5], widths=0.4
)
pdp_plot = plt.boxplot(
    data_to_plot_2, showfliers=False, positions=[1, 3, 5], widths=0.4
)
shap_plot = plt.boxplot(
    data_to_plot_3, showfliers=False, positions=[1.5, 3.5, 5.5], widths=0.4
)

define_box_properties(ale_plot, "#8BC1F7", "Training 1")
define_box_properties(pdp_plot, "#8A8D90", "Training 2")
define_box_properties(shap_plot, "#4CB140", "Training 3")

plt.xticks([1, 3, 5], ["ALE", "PDP", "SHAP"])
plt.xlabel("Method [-]")
plt.ylabel("Runtime [s]")
plt.legend(loc="lower left")

plt.show()
