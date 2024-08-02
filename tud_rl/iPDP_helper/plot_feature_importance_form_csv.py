import pandas as pd
import matplotlib.pyplot as plt
import os


def get_last_folder_name(path):
    return os.path.basename(os.path.normpath(path))


IPDP = False
DIR_PLOTS_IPDP = "/media/jonas/SSD_new/CMS/Semester_5/Masterarbeit/plots/final/iPDP/ipdp_3/2024-07-26_16-01/"


if IPDP == True:
    name_csv = "feature_importance_pdp.csv"

    data = pd.read_csv(os.path.join(DIR_PLOTS_IPDP, name_csv))
    
    columns_to_normalize = data.columns[1:]
    global_min = data[columns_to_normalize].min().min()
    global_max = data[columns_to_normalize].max().max()
    data[columns_to_normalize] = (data[columns_to_normalize] - global_min) / (global_max - global_min)

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
        plt.title(
            r"$\bf{iPDP-based\ Feature\ Importance}$"
            + "\n"
            + rf"exponentially smoothed with $\alpha = \dfrac{{2}}{{{span}+1}}$"
        )

    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(DIR_PLOTS_IPDP, "FI_iPDP_nomalized.pdf")
    )


else:
    DIR_PLOTS = "/media/jonas/SSD_new/CMS/Semester_5/Masterarbeit/plots/final/explanations_4mil-timesteps/plots/explainer_3/2024-07-27_17-50/"


    for dirpath, dirnames, filenames in os.walk(DIR_PLOTS):
        if not dirnames:  # Only consider last-level folders
            last_folder_name = get_last_folder_name(dirpath)
            name_csv = f"feature_importance_{last_folder_name}.csv"

            data = pd.read_csv(os.path.join(DIR_PLOTS, last_folder_name, name_csv))
            
            columns_to_normalize = data.columns[1:]
            global_min = data[columns_to_normalize].min().min()
            global_max = data[columns_to_normalize].max().max()
            data[columns_to_normalize] = (data[columns_to_normalize] - global_min) / (global_max - global_min)

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
                elif last_folder_name == "tree":
                    plt.title(
                        r"$\bf{DecisionTree-based\ Feature\ Importance}$"
                        + "\n"
                        + rf"exponentially smoothed with $\alpha = \dfrac{{2}}{{{span}+1}}$"
                    )

            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(DIR_PLOTS, last_folder_name, f"FI_{last_folder_name}_nomalized.pdf")
            )

            # plt.show()
