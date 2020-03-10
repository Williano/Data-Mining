import os

# Third party library imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def normailze_data(processed_data_file):

    basename_of_processed_data_folder = os.path.basename("processed_data")

    cols = ["column2", "column3", "column4"]
    col = ["column1"]

    data = pd.read_csv(processed_data_file, usecols = cols)
    column1_data = pd.read_csv(processed_data_file, usecols = col)

    scale_range = (0, 1)
    scaler = MinMaxScaler(feature_range=scale_range)
    scaled_data = scaler.fit_transform(data)

    scaled_data_info = pd.DataFrame(scaled_data, columns=cols)

    merged_data = column1_data.join(scaled_data_info)

    scaled_data_info.to_csv(os.path.join(basename_of_processed_data_folder, r'scaled _data.csv'), index=False)
    merged_data.to_csv(os.path.join(basename_of_processed_data_folder, r'normalized_data.csv'), index=False)



def main():


    PROCESSED_FILE = "processed_data.csv"

    normailze_data(PROCESSED_FILE)

if __name__ == "__main__":
    main()