from pyarc import TransactionDB
from pyarc.algorithms import (
    top_rules,
    createCARs,
    M1Algorithm,
    M2Algorithm
)
import pandas as pd


data_train = pd.read_csv("train_data.csv")
data_test = pd.read_csv("train_data.csv")

txns_train = TransactionDB.from_DataFrame(data_train)
txns_test = TransactionDB.from_DataFrame(data_test)

# get the best association rules
rules = top_rules(txns_train.string_representation)
print(rules)

# convert them to class association rules
cars = createCARs(rules)

classifier = M1Algorithm(cars, txns_train).build()
print(classifier)
# classifier = M2Algorithm(cars, txns_train).build()

accuracy = classifier.test_transactions(txns_test)
print(accuracy)


# from pyarc import CBA, TransactionDB
# import pandas as pd
# cba = CBA(support=1.0, confidence=1.0, algorithm="m1")
# cba.fit(txns_train)

# accuracy = cba.rule_model_accuracy(txns_test)
# print(accuracy)