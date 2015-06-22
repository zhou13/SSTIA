from sklearn import svm
import math

# Add your code
svc_x = range(0, len(train_x))
fea_size = 92;

for i in range(0, len(train_x)):
	fea_len = len(x[i])
	svc_x[i] = range(0, fea_size)
	for j in range(0, fea_size):
		svc_x[i][j] = 0
	for j in range(0, fea_len):
		for k in range(0, fea_size):
			svc_x[i][k] += train_x[i][j][k] / fea_len;

clf = svm.SVC();
clf.fit(svc_x, train_y);



