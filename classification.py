#!/usr/bin/python3
from sklearn.externals import joblib
from sklearn import svm
from sklearn import linear_model
from sklearn.multiclass import OneVsOneClassifier
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
import sys
import pickle
import numpy as np

filename = "models/SLR_train_100"
train_ratio = 0.8

print(">>>> Loading")
# pca = joblib.load(filename + ".pca")

kmeans = joblib.load(filename + ".kmeans")
kmeans.verbose = 0

data = pickle.load(open("SLR_train_x.bin", "rb"))
all_y = pickle.load(open("SLR_train_y.bin", "rb"))
print(">>>> Loading finished")

feature_vec = np.zeros((len(data), kmeans.n_clusters))
for i in range(len(data)):
    mydata = data[i]
    # mydata = pca.transform(mydata)
    feature_seq = kmeans.predict(mydata)
    for j in feature_seq:
        feature_vec[i][feature_seq[j]] += 1

feature_vec = normalize(feature_vec)

train_x, test_x, train_y, test_y = \
    train_test_split(feature_vec, all_y, test_size = 1-train_ratio)

print(feature_vec.shape)
print(">>>> Data prepared")

# for alpha_ in [0.1, 0.01, 0.02, 0.03, 0.05, 0.008, 0.009, 0.006, 0.005]:
for alpha_ in [0.0001]:
    clf = OneVsOneClassifier(linear_model.SGDClassifier(alpha = alpha_, n_iter=150000, shuffle=True), n_jobs=4)
    clf.fit(train_x, train_y)
    print("       alpha", alpha_)
    print("       train score", clf.score(train_x, train_y))
    print("       test score",  clf.score(test_x, test_y))
    print(clf)

pred_y = clf.predict(test_x)
print(test_x[:2,:5])
print(pred_y)
