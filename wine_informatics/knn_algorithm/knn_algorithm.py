import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import shuffle


def load_dataset(wine_dataset_csv):
    wine_dataframe = pd.read_csv(wine_dataset_csv, index_col=False)

    return wine_dataframe

def analyse_dataset(processed_data):

    data = processed_data

    print(data.head())


def knn_algorithm_with_holdout_validation(wine_dataset, k_value):

    shuffled_data = wine_dataset.sample(frac=1).reset_index(drop=True)

    label = wine_dataset["Class"].values
    #dataset = wine_dataset.iloc[:, : 486].values
    dataset = wine_dataset.iloc[:, : 485].values
    #dataset = wine_dataset.iloc[:, : 482].values


    X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.20, random_state=1, stratify=label)


    classifier = KNeighborsClassifier(n_neighbors=k_value, algorithm='auto', metric="jaccard")
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Accuracy for {k_value} is:")
    print(accuracy)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print("********************************************")
    # error = []

    # # Calculating error for K values between 1 and 40
    # for i in range(1, 100):
    #     knn = KNeighborsClassifier(n_neighbors=i)
    #     knn.fit(X_train, y_train)
    #     pred_i = knn.predict(X_test)
    #     error.append(np.mean(pred_i != y_test))

    # plt.figure(figsize=(12, 6))
    # plt.plot(range(1, 100), error, color='red', linestyle='dashed', marker='o',
    #         markerfacecolor='blue', markersize=10)
    # plt.title('Error Rate K Value')
    # plt.xlabel('K Value')
    # plt.ylabel('Mean Error')
    # plt.show()


def knn_algorithm_with_k_fold_validation(wine_dataset, k_value):

    #shuffled_data = wine_dataset.sample(frac=1).reset_index(drop=True)

    # Extract features and label
    label = wine_dataset["Class"].values
    #dataset = wine_dataset.iloc[:, : 486].values
    #dataset = wine_dataset.iloc[:, : 485].values
    dataset = wine_dataset.iloc[:, : 482].values

    # Create classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=k_value,  algorithm='auto', metric="jaccard")

    # Train model with 10 fold cross validation
    cross_validation_scores = cross_val_score(knn_classifier, dataset, label, cv=10)

    print(f"The cross validation for {k_value} is: ")
    print(cross_validation_scores)
    print()
    print("Cross validation scores mean: {}%".format(np.mean(cross_validation_scores) * 100))

    print("************************************************************************")

def knn_algorithm_with_hypertuning(wine_dataset):

    #shuffled_data = wine_dataset.sample(frac=1).reset_index(drop=True)

    # Extract features and label
    label = wine_dataset["Class"].values
    dataset = wine_dataset.iloc[:, : 486].values

    # Create classifier
    knn_classifier = KNeighborsClassifier(metric="jaccard")

    #create a dictionary of all values we want to test for n_neighbors
    param_grid = {"n_neighbors": np.arange(1, 100)}

    #use gridsearch to test all values for n_neighbors
    knn_gscv = GridSearchCV(knn_classifier, param_grid, cv=10)

    #fit model to data
    knn_gscv.fit(dataset, label)

    #check top performing n_neighbors value
    best_parameters = knn_gscv.best_params_

    #check mean score for the top performing value of n_neighbors
    best_score = knn_gscv.best_score_ * 100

    print(best_parameters)
    print()
    print(best_score)

def main():

    #wine_dataset_file = "drink_and_hold_dataset.csv"

    #tweaked_wine_dataset_file = "drink_and_hold_dataset_with_finish_attribute_deleted.csv"

    tweaked_wine_dataset_file = "drink_and_hold_dataset_with_4_attributes_above_35_percent_deleted.csv"

    processed_data_file = load_dataset(tweaked_wine_dataset_file)

    #analyse_dataset(processed_data_file)

    #knn_algorithm_with_holdout_validation(processed_data_file)

    #knn_algorithm_with_k_fold_validation(processed_data_file)

    #knn_algorithm_with_hypertuning(processed_data_file)

    k_values = [20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 77, 85, 95, 100]

    for k in k_values:
          knn_algorithm_with_k_fold_validation(processed_data_file, k)



if __name__ == "__main__":
    main()