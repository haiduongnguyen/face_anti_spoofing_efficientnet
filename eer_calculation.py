import numpy as np

from scipy import interpolate

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def cal_metric(groundTruth, predicted):
	fpr, tpr, thresholds = roc_curve(groundTruth, predicted)
	y = (tpr)
	x = (fpr)
	z = tpr +fpr
	tpr = tpr.reshape((tpr.shape[0],1))
	fpr = fpr.reshape((fpr.shape[0],1))
	xnew = np.arange(0, 1, 0.0000001)
	func = interpolate.interp1d(x, y)

	ynew = func(xnew)

	znew = abs(xnew + ynew-1)

	eer = xnew[np.argmin(znew)]

	h_new = abs(eer - fpr)
	# print(h_new)
	# print(thresholds)
	eer_threshold = thresholds[np.argmin(h_new)]

#	print('EER',eer)

	FPR = {"TPR(1.%)": 0.01, "TPR(.5%)": 0.005, "TPR(5%)" : 0.05, "TPR(10%)" : 0.1}


	TPRs = {"TPR(1.%)": 0, "TPR(.5%)": 0, "TPR(5%)" : 0, "TPR(10%)" : 0}
	for i, (key, value) in enumerate(FPR.items()):

		index = np.argwhere(xnew == value)

		score = ynew[index] 

		TPRs[key] = float(np.squeeze(score))
#	    print(key, score)
	if 1:
		plt.figure("auc")
		plt.plot(xnew, ynew, label = "auc")
		plt.xlabel("spoof_threshold")
		plt.ylabel(" ")

		plt.figure("fnr and fpr")
		plt.plot( thresholds, 1 - tpr, label="fnr")
		plt.plot(thresholds, fpr, label = "fpr")
		plt.xlabel("spoof_threshold")
		plt.ylabel(" ")

		plt.legend()
		plt.show()
	auc = roc_auc_score(groundTruth, predicted)
	return eer,TPRs, auc, {'x':xnew, 'y':ynew}, eer_threshold
