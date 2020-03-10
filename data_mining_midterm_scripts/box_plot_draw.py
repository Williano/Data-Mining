import os

# Third party library imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def draw_box_plot_with_seaborn(normalized_data_file):

    basename_of_processed_data_folder = os.path.basename("processed_data")

    col = ["column3"]
    data = pd.read_csv(normalized_data_file)
    sns.set_style("whitegrid")
    sns.boxplot(y="column3", data=data, orient="v")
    plt.show()

def draw_box_plot_with_matplotlib(normalized_data_file):

    col = ["column3"]
    data = pd.read_csv(normalized_data_file)
    data.boxplot(column=col)
    plt.show()


def main():


    NORMALIZED_FILE = "normalized_data.csv"

    draw_box_plot_with_matplotlib(NORMALIZED_FILE)
    draw_box_plot_with_seaborn(NORMALIZED_FILE)

if __name__ == "__main__":
    main()