# Standard library imports
import os

def create_folder_to_store_processed_data():

    if not os.path.exists("processed_data"):
        path = os.makedirs("processed_data")


def main():

    create_folder_to_store_processed_data()

if __name__ == "__main__":
    main()