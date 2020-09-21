import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import shuffle


def load_dataset(wine_dataset_csv):
    wine_dataframe = pd.read_csv(wine_dataset_csv, index_col=False)

    return wine_dataframe

def analyse_dataset(processed_data):

    data = processed_data

    print(data.head())


def knn_algorithm_with_holdout_validation(wine_dataset):

    # shuffled_data = shuffle(wine_dataset)
    # shuffled_data.reset_index(inplace=True, drop=True)

    label = wine_dataset["Class"].values
    dataset = wine_dataset.iloc[:, : 486].values


    X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.20, random_state=1, stratify=label)


    classifier = KNeighborsClassifier(n_neighbors=37)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred) * 100
    print(accuracy)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # error = []

    # # Calculating error for K values between 1 and 40
    # for i in range(1, 60):
    #     knn = KNeighborsClassifier(n_neighbors=i)
    #     knn.fit(X_train, y_train)
    #     pred_i = knn.predict(X_test)
    #     error.append(np.mean(pred_i != y_test))

    # plt.figure(figsize=(12, 6))
    # plt.plot(range(1, 60), error, color='red', linestyle='dashed', marker='o',
    #         markerfacecolor='blue', markersize=10)
    # plt.title('Error Rate K Value')
    # plt.xlabel('K Value')
    # plt.ylabel('Mean Error')
    # plt.show()


def knn_algorithm_with_k_fold_validation(wine_dataset):

    shuffled_data = shuffle(wine_dataset)
    shuffled_data.reset_index(inplace=True, drop=True)

    # Extract features and label
    label = shuffled_data["Class"].values
    dataset = shuffled_data.iloc[:, : 486].values

    # Create classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=30)

    # Train model with 10 fold cross validation
    cross_validation_scores = cross_val_score(knn_classifier, dataset, label, cv=10)


    print(cross_validation_scores)
    print("cross validation scores mean:{}".format(np.mean(cross_validation_scores)))


def main():

    wine_dataset_file = "drink_and_hold_dataset.csv"

    processed_data_file = load_dataset(wine_dataset_file)

    #analyse_dataset(processed_data_file)

    #knn_algorithm_with_holdout_validation(processed_data_file)

    knn_algorithm_with_k_fold_validation(processed_data_file)




if __name__ == "__main__":
    main()