#!/usr/bin/python3
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import scale
import sys
import itertools
import pickle
import numpy as np
import gc

PCA_COMP = 50
JOBS = 8

print(">>>> Loading", sys.argv[1], "with", sys.argv[2], "clusters in", sys.argv[3])
data = pickle.load(open(sys.argv[1], "rb"))
print(">>>> Loading finished")

# print(">>>> PCAing")
# print("       data shape:", data.shape)
# pca = RandomizedPCA(n_components=PCA_COMP, copy=False, whiten=True)
# data = pca.fit_transform(data)
# # print("       PCA variance:", pca.explained_variance_ratio_)
# joblib.dump(pca, sys.argv[3]+".pca")
# print(">>>> PCA finished")

print(">>>> Kmeans begin")
data = np.vstack(data)
data = shuffle(data)
data = scale(data)
print("       data shape:", data.shape)
# kmeans = KMeans(n_clusters=int(sys.argv[2]), n_init=JOBS, n_jobs=8, verbose=1).fit(data)
kmeans = MiniBatchKMeans(
    n_clusters=int(sys.argv[2]),
    n_init=JOBS,
    max_no_improvement=5000,
    reassignment_ratio=20 / int(sys.argv[2]) * 0.01,
    batch_size=500).fit(data)
print("       inertia:", kmeans.inertia_)
joblib.dump(kmeans, sys.argv[3]+".kmeans")
