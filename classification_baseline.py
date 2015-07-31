#!/usr/bin/python3
from sklearn import linear_model
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.utils import shuffle
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
import math
import pickle
import numpy as np

train_ratio = 0.9

print(">>>> Loading")
all_x_pre = pickle.load(open("SLR_train_x.bin", "rb"))
all_y = pickle.load(open("SLR_train_y.bin", "rb"))
fea_size = len(all_x_pre[0][0])
print(">>>> Loading finished")
all_x = np.zeros((len(all_x_pre), fea_size))
print(all_x.shape)

for i, arr in enumerate(all_x_pre):
    sums = np.zeros((fea_size,))
    for time_feature in arr:
        sums += time_feature
    sums /= len(arr)
    all_x[i] = sums

# train_x, test_x, train_y, test_y = \
#     train_test_split(all_x, all_y, test_size = 1-train_ratio)
sep = int(len(all_x) * train_ratio)
train_x = all_x[:sep]
train_y = all_y[:sep]
test_x = all_x[sep:]
test_y = all_y[sep:]
train_x, train_y = shuffle(train_x, train_y)

print(">>>> Data prepared")

print(">>>> Training Model")
tuned_parameters = [
    {
        'kernel': ['rbf'],
        'C': [1, 3, 10,  50, 100],
        'gamma': [0.003, 0.0005],
    },
]

# clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, n_jobs=8)
# for c in [5, 10, 20, 50, 100]:
#     for gamma_ in [0.01, 0.03, 0.001, 0.003, 0.0001]:
#         clf = svm.SVC(kernel='rbf', C=c, gamma=gamma_)
# clf = AdaBoostClassifier(n_estimators=1000)
# clf = GradientBoostingClassifier(n_estimators=100)
clf = ExtraTreesClassifier(n_estimators=1000, max_depth=8)
# clf = svm.LinearSVC(max_iter=8000)
# clf = OneVsOneClassifier(linear_model.SGDClassifier(n_iter=25000, shuffle=True), n_jobs=8)
clf.fit(train_x, train_y)
print("       train score", clf.score(train_x, train_y))
print("       test score",  clf.score(test_x, test_y))
print(clf)


# print("Best parameters set found on development set:")
# print()
# print(clf.best_params_)
# print()
# print("Grid scores on development set:")
# print()
# for params, mean_score, scores in clf.grid_scores_:
#     print("%0.3f (+/-%0.03f) for %r"
#           % (mean_score, scores.std() * 2, params))
# print()

# pred_y = clf.predict(test_x)
# print(len(train_x), len(test_x))
# print(test_y)
# print(pred_y)
# print("       Accuracy:", accuracy_score(test_y, pred_y))
