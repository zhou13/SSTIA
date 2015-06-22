#!/usr/bin/python3
from sklearn import svm
from sklearn import linear_model
from sklearn.multiclass import OneVsOneClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import math
import pickle
import numpy as np

fea_size = 92
train_ratio = 0.8

print(">>>> Loading")
all_x_pre = pickle.load(open("SLR_train_x.bin", "rb"))
all_y = pickle.load(open("SLR_train_y.bin", "rb"))

print(">>>> Loading finished")
all_x = np.zeros((len(all_x_pre), fea_size))

for i, arr in enumerate(all_x_pre):
    sums = np.zeros((fea_size,))
    for time_feature in arr:
        sums += time_feature
    sums /= len(arr)
    all_x[i] = sums

all_x, all_y = shuffle(all_x, all_y, random_state=5)

sep = int(len(all_x) * train_ratio)
train_x = all_x[:sep]
train_y = all_y[:sep]
test_x = all_x[sep:]
test_y = all_y[sep:]

print(">>>> Data prepared")

# clf = svm.SVC()
# clf = svm.LinearSVC(max_iter=8000)
clf = OneVsOneClassifier(linear_model.SGDClassifier(n_iter=50000, shuffle=True), n_jobs=4)
clf.fit(train_x, train_y)
print("       train score", clf.score(train_x, train_y))
print("       test score",  clf.score(test_x, test_y))
print(clf)

pred_y = clf.predict(test_x)
# print(len(train_x), len(test_x))
print(test_x[:2,:5])
# print(test_y)
print(pred_y)
# print("       Accuracy:", accuracy_score(test_y, pred_y))
