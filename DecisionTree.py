import numpy as np
from sklearn import tree, preprocessing
import csv
from calf1 import caculateF1
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

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
results = np.array_split(result, 10)
print("results: ", results)
accuracies = []
precisions = []
recalls = []
F1s = []
for i in range(len(results)):
    print("start with test set: ", i)
    test_set = results[i]
    train_sets = results[:i] + results[i+1:]
    train_set = train_sets[0]
    j = 1
    while j < 9:
        train_set = np.concatenate((train_set, train_sets[j]))
        j = j + 1
    test_x = test_set[:, 0:-1]
    test_y = test_set[:, -1]
    train_x = train_set[:, 0:-1]
    train_y = train_set[:, -1]
    clf = tree.DecisionTreeClassifier(max_depth=8)
    clf.fit(train_x, train_y)
    predict_y = clf.predict(test_x)
    # precision, recall, F1 = caculateF1(test_y, predict_y)
    # accu = predict_y == test_y
    # accu = 1 * accu
    # accu = np.sum(accu) / len(accu)
    precision = precision_score(test_y, predict_y, average='macro')
    accu = accuracy_score(test_y, predict_y)
    recall = recall_score(test_y, predict_y, average='macro')
    F1 = (2.0 * recall * precision) / (recall + precision)
    print("Accuracy: ", accu)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", F1)
    print("---------------------------------------------")
    accuracies.append(accu)
    precisions.append(precision)
    recalls.append(recall)
    F1s.append(F1)
print("Final Average Accuracy: ", np.average(accuracies))
print("Final Average Precisions: ", np.average(precisions))
print("Final Average Recalls: ", np.average(recalls))
print("Final Average F1s: ", np.average(F1s))
# train_x = x[0:90000]
# train_y = y[0:90000]
# test_x = x[90000:]
# test_y = y[90000:]


# for i in range(20):
#     clf = tree.DecisionTreeClassifier(max_depth=i+1)
#     clf.fit(train_x, train_y)
#     print(train_y)
#     accu = clf.predict(test_x) == test_y
#     accu = 1 * accu
#     print("Accuracy: ", np.sum(accu) / len(accu), "|| with depth: ", i+1)
# print("Accuracy: ", accu.astype())
