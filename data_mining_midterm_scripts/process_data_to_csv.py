# Standard Library imports
import os

# Third party library imports
import pandas as pd

# Data Mining program imports
from utility_functions import create_folder_to_store_processed_data

def read_text_file(text_file):

    file_data = pd.read_csv(text_file, delimiter="\t")

    return file_data


def process_text_file(read_file_data):

    # absolute_path_of_processed_data_folder = os.path.abspath("processed_data")
    # relative_path_of_processed_data_folder = os.path.realpath("processed_data")

    basename_of_processed_data_folder = os.path.basename("processed_data")

    row_data = [
          {
            "column1": row[1][0],
            "column2": row[1][1],
            "column3": row[1][2],
            "column4": row[1][3],
        }
        for row in read_file_data.iterrows()
        ]

    file_info = pd.DataFrame.from_dict(row_data)
    file_info.to_csv(os.path.join(basename_of_processed_data_folder, r'processed_data.csv'), index=False)


def main():

    RAW_TEXT_FILE = "MidTerm_exam_input_file.txt"

    create_folder_to_store_processed_data()

    processed_file = read_text_file(RAW_TEXT_FILE)

    process_text_file(processed_file)

if __name__ == "__main__":
    main()