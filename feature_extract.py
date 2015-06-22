#!/usr/bin/python3
from features import *
import scipy.io.wavfile as wav
import numpy as np
import pickle

train_x = []
train_y = []
test_x = []

with open("SLR/TRAIN_INFO.txt", "r") as ftrain:
    ftrain.readline()
    language_index = {}
    total_index = 0
    for iline, line in enumerate(ftrain):
        filename, language = line.split()
        if language not in language_index:
            language_index[language] = total_index
            total_index += 1
        train_y.append(language_index[language])

        (rate,sig) = wav.read("SLR/" + filename)
        mfcc_feat = mfcc(sig,rate)
        fbank_feat = fbank(sig,rate)
        fbank_feat0 = fbank(sig,rate)[0]
        fbank_feat1 = fbank(sig,rate)[1]
        logfbank_feat = logfbank(sig,rate)
        ssc_feat = ssc(sig, rate)
        feat = np.c_[mfcc_feat, fbank_feat0, fbank_feat1, logfbank_feat, ssc_feat]
        print(iline, ":", feat.shape)
        train_x.append(feat)

with open("SLR_train_x.bin", "wb") as fout:
    pickle.dump(train_x, fout)
with open("SLR_train_y.bin", "wb") as fout:
    pickle.dump(train_y, fout)
