import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt

def normalize(data):
	res = (data - np.mean(data))/np.std(data)

	return res

def detecData(observer):

	foldername = '..\\detection\\'+observer
	files = os.listdir(foldername)

	y = []
	X = []
	N = len(files) # number of data segments
	for i in range(1): 
		fn = files[i]
		mat = scipy.io.loadmat(foldername+'\\'+fn)
		k = list(mat.keys())
		if len(k) == 6:
			y.append(0)
		else:
			y.append(1)

		D = mat[k[3]]
		mean = np.mean(D,axis = 1)
		maximum = np.amax(D,axis = 1)
		minimum = np.amin(D,axis = 1)
		stdv = np.std(D,axis = 1)

		# normalize
		mean = normalize(mean)
		maximum = normalize(maximum)
		minimum = normalize(minimum)
		stdv = normalize(stdv)

	return 0

def checkD():
	foldn = '..\\prediction\\Dog_1'

	files = os.listdir(foldn)

	# prediction
	# ['__header__', '__version__', '__globals__', 'interictal_segment_1']
	mat = scipy.io.loadmat(foldn+'\\'+files[0])
	k = list(mat.keys())
	print(k)
	print(mat[k[3]][0][0][0].shape)

	foldn = '..\\detection\\Dog_1'
	files = os.listdir(foldn)

	# detection
	# ['__header__', '__version__', '__globals__', 'data', 'freq', 'channels', 'latency'] -- ictal
	# OR
	# ['__header__', '__version__', '__globals__', 'data', 'freq', 'channels'] -- interictal
	mat = scipy.io.loadmat(foldn+'\\'+files[2000])
	k = list(mat.keys()) # 
	print(k)
	print(mat[k[3]].shape)


detecData('Dog_1')