import pandas as pd
import matplotlib.pyplot as plt
import os


def get_last_folder_name(path):
    return os.path.basename(os.path.normpath(path))


DIR_PLOTS = "/media/jonas/SSD_new/CMS/Semester_5/Masterarbeit/plots/ple_ale_SHAP_interrupted/background-300_explain_1000/plots/2024-07-22_21-32/"

for dirpath, dirnames, filenames in os.walk(DIR_PLOTS):
    if not dirnames:  # Only consider last-level folders
        last_folder_name = get_last_folder_name(dirpath)
        name_csv = f"feature_importance_{last_folder_name}.csv"

        data = pd.read_csv(os.path.join(DIR_PLOTS, last_folder_name, name_csv))

        # exponential moving average
        span = 3  # 1 for no smoothing
        data = data.ewm(span=span, adjust=False).mean()

        plt.figure(figsize=(12, 8))
        for column in data.columns[1:]:
            plt.plot(data["time_step"], data[column], label=column, linewidth=3.0)

        plt.xlabel("Time Step")
        plt.ylabel("Feature Importance")
        if span == 1:
            plt.title(r"$\bf{PDP-based\ Feature\ Importance}$" + "\nwithout smoothing")
        else:
            if last_folder_name == "pdp":
                plt.title(
                    r"$\bf{PDP-based\ Feature\ Importance}$"
                    + "\n"
                    + rf"exponentially smoothed with $\alpha = \dfrac{{2}}{{{span}+1}}$"
                )
            elif last_folder_name == "ale":
                plt.title(
                    r"$\bf{ALE-based\ Feature\ Importance}$"
                    + "\n"
                    + rf"exponentially smoothed with $\alpha = \dfrac{{2}}{{{span}+1}}$"
                )
            elif last_folder_name == "SHAP":
                plt.title(
                    r"$\bf{SHAP-based\ Feature\ Importance}$"
                    + "\n"
                    + rf"exponentially smoothed with $\alpha = \dfrac{{2}}{{{span}+1}}$"
                )

        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(DIR_PLOTS, last_folder_name, f"FI_{last_folder_name}.pdf")
        )

        # plt.show()
