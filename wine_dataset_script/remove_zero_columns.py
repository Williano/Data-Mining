import pandas as pd
import csv



def load_wine_dataset_sheet(wine_data_set_file):
    """
        Returns the wine dataset workbook file.

        Parameter:
        wine_data_set_file: The excel file containing the wine dataset.

        Returns:
        workbook_file: The workbook of the excel wine dataset.
    """

    # Load the Excel document
    workbook_file = pd.read_excel(wine_data_set_file, "Sheet1")

    return workbook_file

def remove_columns_with_zeros(wine_dataset_file):

    cleaned_drink_duration_dataset_data = wine_dataset_file.loc[:,
    (wine_dataset_file !=0).any(axis=0)]

    cleaned_drink_duration_dataset_data.to_excel(
        "cleaned_drink_duration_dataset.xlsx", index=False)

    cleaned_drink_duration_dataset_data.to_csv(
        "cleaned_drink_duration_dataset.csv", index=False)



def main():
    """
       Defines the main entry point of the script.
    """

    # Declare a constant to store the excel file name.
    WINE_DATA_SET_FILE = "DrinkDurationDataset.xlsx"

    # Loads and read file from excel
    data_sheet = load_wine_dataset_sheet(WINE_DATA_SET_FILE)

    # Cleans the loaded file
    remove_columns_with_zeros(data_sheet)

# Check if the file name is main, if the name is main run the script else do not run the script.
if __name__ == "__main__":
    main()