import numpy as np
import pandas as pd
from sklearn import tree, preprocessing
import csv

reader = csv.reader(open("complete.csv", "rt"), delimiter=",")
result = list(reader)
result = np.array(result)
result = result[1:, 2:]
permutation = list(np.random.permutation(len(result)))
result = result[permutation, :]
le = preprocessing.LabelEncoder()
for i in range(result.shape[1]):
    result[:, i] = le.fit_transform(result[:, i])
print(result.shape)
print(result)
x = result[:, 0:-1]
y = result[:, -1]
train_x = x[0:90000]
train_y = y[0:90000]
test_x = x[90000:]
test_y = y[90000:]
for i in range(20):
    clf = tree.DecisionTreeClassifier(max_depth=i+1)
    clf.fit(train_x, train_y)
    accu = clf.predict(test_x) == test_y
    accu = 1 * accu
    print("Accuracy: ", np.sum(accu) / len(accu), "|| with depth: ", i+1)
# print("Accuracy: ", accu.astype())
