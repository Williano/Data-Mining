import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle


def load_dataset(wine_dataset_csv):
    wine_dataframe = pd.read_csv(wine_dataset_csv, index_col=False)

    return wine_dataframe

def analyse_dataset(processed_data):

    data = processed_data

    print(data.head())


def knn_algorithm(wine_dataset):

    shuffled_data = shuffle(wine_dataset)
    shuffled_data.reset_index(inplace=True, drop=True)

    label = shuffled_data["Class"].values
    dataset = shuffled_data.iloc[:, : 486].values


    X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.20)


    classifier = KNeighborsClassifier(n_neighbors=30)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    error = []

    # Calculating error for K values between 1 and 40
    for i in range(1, 100):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 100), error, color='red', linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()


def main():

    wine_dataset_file = "drink_and_hold_dataset.csv"

    processed_data_file = load_dataset(wine_dataset_file)

    #analyse_dataset(processed_data_file)

    knn_algorithm(processed_data_file)




if __name__ == "__main__":
    main()