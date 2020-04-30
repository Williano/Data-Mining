from read import read
from pre_processing import pre_process
from cba_rg import rule_generator
from cba_cb_m1 import classifier_builder_m1
import random


# calculate the error rate of the classifier on the dataset
def get_error_rate(classifier, dataset):
    size = len(dataset)
    error_number = 0
    for case in dataset:
        is_satisfy_value = False
        for rule in classifier.rule_list:
            is_satisfy_value = is_satisfy(case, rule)
            if is_satisfy_value == True:
                break
        if is_satisfy_value == False:
            if classifier.default_class != case[-1]:
                error_number += 1
    return error_number / size

def main():

    test_data_path = 'train.data.csv'
    test_scheme_path = 'wine.names.csv'

    # test_data_path = 'datasets/iris.data'
    # test_scheme_path = 'datasets/iris.names'

    data, attributes, value_type = read(test_data_path, test_scheme_path)
    random.shuffle(data)
    train_dataset = pre_process(data, attributes, value_type)

    cars = rule_generator(train_dataset, 0.22, 0.6)
    cars.prune_rules(train_dataset)
    cars.rules = cars.pruned_rules

    classifier_m1 = classifier_builder_m1(cars, train_dataset)


    # error_rate = get_error_rate(classifier_m1, train_dataset)

    total_car_number = len(cars.rules)
    # total_classifier_rule_num = len(classifier_m1.rule_list)

    # print("_______________________________________________________")
    # print(error_rate)

    # print("_______________________________________________________")
    # print(total_classifier_rule_num)

    print("_______________________________________________________")
    cars.print_rule()
    print("_______________________________________________________")
    cars.prune_rules(train_dataset)
    cars.print_pruned_rule()
    print("_______________________________________________________")
    print()
    classifier_m1.print()

    print("_______________________________________________________")
    print(total_car_number)

main()
