# Standard Library imports
import csv
from itertools import islice

# Third-party library imports
import pandas as pd
import numpy as np



def get_wine_attributes(wine_dataset_file):

    #wine_details = wine_dataset_file.loc[:, "Wine"]
    wine_attributes = wine_dataset_file.iloc[:, 10:495]

    wine_attribute_list = list()

    for wine_attribute in wine_attributes.itertuples(index=False, name="Wine"):
       attribute = wine_attribute._asdict()

       #att1 = {"att" :[ key for key,value in attribute.items() if value == 1]}

       non_zero_attirbutes = [ str(key).replace('_','') for key,value in attribute.items() if value == 1]
       wine_attribute_list.append(non_zero_attirbutes)

    wine_attribute_list_dataframe = pd.DataFrame(wine_attribute_list)

    #df.index = np.arange(1,len(df)+1)

    # Drinkable
    # wine_attribute_list_dataframe.to_csv("hold_wine_attributes_numeric.csv", header=False, sep=" ", index=False)

    # Undrinkable
    wine_attribute_list_dataframe.to_csv("drink_wine_attributes_numeric.csv", header=False, sep=" ", index=False)



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


def main():
    """
       Defines the main entry point of the script.
    """

    # Declare a constant to store the excel file name.
    # HOLD_WINE_DATASET_FILE = "hold_data.xlsx"

    # # Loads and read file from excel
    # hold_data_sheet = load_wine_dataset_sheet(HOLD_WINE_DATASET_FILE)

    # # Gets hold Wine attributes
    # get_wine_attributes(hold_data_sheet)


    #Declare a constant to store the excel file name.
    DRINK_WINE_DATASET_FILE = "drink_data.xlsx"

    # Loads and read file from excel
    drink_data_sheet = load_wine_dataset_sheet(Drink_WINE_DATASET_FILE)

    # Gets drinkable Wine attributes
    get_wine_attributes(drink_data_sheet)


# Check if the file name is main, if the name is main run the script else do not run the script.
if __name__ == "__main__":
    main()