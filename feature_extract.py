#!/usr/bin/python3
from features import *
import scipy.io.wavfile as wav
import numpy as np
import pickle
import gc
import sys

SAMPLE_LENGTH = 5

slr_train_x = []
slr_train_y = []
slr_test_x = []
slc_data = []

def delta(mat):
    mat1 = np.roll(mat, shift=+1, axis=0)
    mat2 = np.roll(mat, shift=-1, axis=0)
    return (mat1 - mat2) / 2

def delta_delta(mat):
    mat1 = np.roll(mat, shift=+1, axis=0)
    mat2 = np.roll(mat, shift=-1, axis=0)
    return (mat1 - 2 * mat + mat2) / 4

def get_feature(filename):
    print(filename)
    rate, sig = wav.read(filename)
    ret = []
    wavelen = rate * SAMPLE_LENGTH
    print(len(sig), wavelen)
    for i in range(len(sig) // wavelen):
        mfcc_feat = mfcc(sig[i*wavelen:(i+1)*wavelen], rate)
        ssc_feat = ssc(sig[i*wavelen:(i+1)*wavelen], rate)
        logfbank_feat = logfbank(sig[i*wavelen:(i+1)*wavelen], rate)
        feat = np.c_[mfcc_feat, ssc_feat, logfbank_feat]
        feat = np.c_[feat, delta(feat), delta_delta(feat)]
        ret.append(feat)

    return ret


with open("SLR/TRAIN_INFO.txt", "r") as ftrain:
    ftrain.readline()
    language_index = {}
    total_index = 0
    for iline, line in enumerate(ftrain):
        filename, language = line.split()

        feat = get_feature("SLR/" + filename)
        print("SLR_TRAIN", iline)
        slr_train_x += feat

        if language not in language_index:
            language_index[language] = total_index
            total_index += 1
        slr_train_y += len(feat) * [language_index[language]]


    with open("SLR_train_x.bin", "wb") as fout:
        pickle.dump(slr_train_x, fout)
    with open("SLR_train_y.bin", "wb") as fout:
        pickle.dump(slr_train_y, fout)

sys.exit(0)

with open("SLR/SLR_TEST.list", "r") as ftest:
    for i, line in enumerate(ftest):
        filename = line.strip()
        feat = get_feature(filename)
        slr_test_x += feat
        print("SLR_TEST", i)
    with open("SLR_test_x.bin", "wb") as fout:
        pickle.dump(slr_test_x, fout)

with open("SLC/SLC_Data.list", "r") as ftest:
    for i, line in enumerate(ftest):
        filename = line.strip()
        feat = get_feature(filename)
        slc_data += feat
        print("SLC", i)
    with open("SLC_data.bin", "wb") as fout:
        pickle.dump(slc_data, fout)
