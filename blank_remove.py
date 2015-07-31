#!/usr/bin/python3
import sys
import scipy.io.wavfile as wav
import numpy as np

filename = sys.argv[1]
# threshold = float(sys.argv[2])

rate, sig = wav.read(filename)
sig = sig[:,0]
new_sig = np.zeros(sig.shape)
csum = np.cumsum(np.abs(sig))
for i in range(len(sig)):
    new_sig[i] = (csum[i] - csum[max(0, i-rate)]) / rate

ans = []
for old, new in zip(sig, new_sig):
    if new > 200:
        ans.append(old)
print(sig)
print(len(sig))
print(len(ans))
wav.write("out.wav", rate, np.array(ans))
