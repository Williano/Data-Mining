import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import shuffle


def load_dataset(wine_dataset_csv):
    wine_dataframe = pd.read_csv(wine_dataset_csv, index_col=False)

    return wine_dataframe


def analyse_dataset(processed_data):

    data = processed_data

    print(data.head())


def naive_bayes_algorithm_with_holdout_validation(wine_dataset):

    # shuffled_data = shuffle(wine_dataset)
    # shuffled_data.reset_index(inplace=True, drop=True)

    label = wine_dataset["Class"].values
    dataset = wine_dataset.iloc[:, : 486].values

    X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.20, random_state=1)

    naive_bayes_classifier = GaussianNB()
    naive_bayes_classifier.fit(X_train, y_train)

    y_pred = naive_bayes_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred) * 100
    print(accuracy)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def naive_bayes_algorithm_with_k_fold_validation(wine_dataset):

    # shuffled_data = shuffle(wine_dataset)
    # shuffled_data.reset_index(inplace=True, drop=True)

    # Extract features and label
    label = wine_dataset["Class"].values
    # dataset = wine_dataset.iloc[:, : 486].values
    # dataset = wine_dataset.iloc[:, : 485].values
    dataset = wine_dataset.iloc[:, : 482].values

    # Create classifier
    naive_bayes_classifier = GaussianNB()

    # Train model with 10 fold cross validation
    cross_validation_scores = cross_val_score(naive_bayes_classifier, dataset,
                                             label, cv=10)

    print(cross_validation_scores)
    print()
    print("Cross validation scores mean: {}%".format(
        np.mean(cross_validation_scores) * 100))


def main():

    wine_dataset_file = "drink_and_hold_dataset.csv"

    # tweaked_wine_dataset_file = "drink_and_hold_dataset_with_finish_attribute_deleted.csv"

    tweaked_wine_dataset_file =\
    "drink_and_hold_dataset_with_4_attributes_above_35_percent_deleted.csv"

    processed_data_file = load_dataset(tweaked_wine_dataset_file)

    # analyse_dataset(processed_data_file)

    # naive_bayes_algorithm_with_holdout_validation(processed_data_file)

    naive_bayes_algorithm_with_k_fold_validation(processed_data_file)


if __name__ == "__main__":
    main()
