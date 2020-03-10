import pandas as pd
import csv

#MINIMUM_YEAR = 5
MINIMUM_YEAR = 6
#MINIMUM_YEAR = 7

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


def check_if_wine_is_drinkable_or_not(wine_data_sheet):

    """
        Checks if a wine is drinkable or not.

        Reads an excel sheet containing a dataset of wines and extract the ones which are
        drinkable for 5 years or more into a new excel file, the ones below 5 years into
        another excel file and the one with no data on when to drink them into another excel file.

        Parameter:
        wine_data_set_file: The excel file containing the wine dataset.

        Returns:
        None
    """

    drinkable_wine_list = list()
    undrinkable_wine_list = list()
    no_wine_data_list = list()
    wine_dict = dict()

    for wine in wine_data_sheet.itertuples(index=False, name="Wine"):

        wine_dict = wine._asdict()

        drinkable_from = wine_dict["From"]
        manufacutred_year = wine_dict["Year"]

        if isinstance(drinkable_from, int):

            years_to_decide_to_drink_or_not = drinkable_from - manufacutred_year

            if  years_to_decide_to_drink_or_not >= MINIMUM_YEAR:

                wine_data = {
                    "Years To Decide To Drink or Not": years_to_decide_to_drink_or_not
                }
                wine_dict.update(wine_data)
                drinkable_wine_list.append(wine_dict)


            elif years_to_decide_to_drink_or_not < MINIMUM_YEAR:
                wine_data = {
                    "Years To Decide To Drink or Not": years_to_decide_to_drink_or_not
                }
                wine_dict.update(wine_data)
                undrinkable_wine_list.append(wine_dict)

            else:
                pass
        else:

            wine_data = {
                    "Year": manufacutred_year,
                }
            wine_dict.update(wine_data)
            no_wine_data_list.append(wine_dict)

    # Calls write to excel and csv function to write the data to excel and csv.
    write_to_excel_and_csv(drinkable_wine_list, undrinkable_wine_list, no_wine_data_list)

def write_to_excel_and_csv(drinkable_list, undrinkable_list, no_data_list):
    """
        Writes list of drinkable, undrinkable and no_wine data to a new excel file.

        Parameter:
        drinkable_list: List containing a dictionary of wines that are drinkable.
        undrinkable_list: List containing a dictionary of wines that are not drinkable.
        no_wine_data: List containing a dictionary of wines that should be drink now.

        Returns:
        None
    """

    drinkable_data = pd.DataFrame.from_dict(drinkable_list)
    undrinkable_data = pd.DataFrame.from_dict(undrinkable_list)
    no_data = pd.DataFrame.from_dict(no_data_list)

    drinkable_data.to_excel('drinkable.xlsx', index=False)
    drinkable_data.to_csv("drinkable.csv", index=False)
    undrinkable_data.to_excel("undrinkable.xlsx", index=False)
    undrinkable_data.to_csv("undrinkable.csv", index=False)
    no_data.to_excel("no_data.xlsx", index=False)
    no_data.to_csv("no_data.csv", index=False)

def main():
    """
       Defines the main entry point of the script.
    """

        # Declare a constant to store the excel file name.
    CLEANED_WINE_DATASET_FILE = "cleaned_drink_duration_dataset.xlsx"


    # Loads and read file from excel
    cleaned_data_sheet = load_wine_dataset_sheet(CLEANED_WINE_DATASET_FILE)

    # Determines which wines are drinkable
    check_if_wine_is_drinkable_or_not(cleaned_data_sheet)


# Check if the file name is main, if the name is main run the script else do not run the script.
if __name__ == "__main__":
    main()