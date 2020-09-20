import pandas as pd



def load_dataset(wine_dataset_csv):
    wine_dataset = pd.read_csv(wine_dataset_csv, index_col=False)

    return wine_dataset

def analyse_dataset(processed_data):

    data = processed_data

    print(data.head())


def knn_algorithm():
    pass

def main():

    wine_dataset_file = "drink_and_hold_dataset.csv"

    processed_data_file = load_dataset(wine_dataset_file)

    analyse_dataset(processed_data_file)


if __name__ == "__main__":
    main()