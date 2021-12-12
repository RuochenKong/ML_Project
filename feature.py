import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pyeeg

def head_generate(channelNum,detec = 1):
	fh = ''
	'''
	for i in range(channelNum*2):
		fh+='fft_%2d,'%(i+1)
	'''
	bh = ''
	th = ''
	meh = ''
	mmh = ''
	for i in range(channelNum):
		meh += 'mean_C%2d,'%(i+1)
		mmh += 'max-min_C%2d,'%(i+1)
		bh += 'delta_C%2d,'%(i+1)+'theta_C%2d,'%(i+1)+'alpha_C%2d,'%(i+1)\
		+'beta_C%2d,'%(i+1)+'low_gamma_C%2d,'%(i+1)+'high_gamma_C%2d,'%(i+1)
		th += 'hjorth_mobility_C%2d,'%(i+1)+'hjorth_complexity_C%2d,'%(i+1)
		if detec == 1:
			th += 'hfd_C%2d,'%(i+1)
		th +='pfd_C%2d'%(i+1)
		if i != channelNum-1:
			th += ','

	res = '1,latency,' + meh + mmh + fh + bh + th

	return res

def normalization(Data):
	meand = np.mean(Data)
	stdd = np.std(Data)

	return (Data - meand)/stdd

def extract(Data,sample_freq,channelNum,time,detec = 1):
	mean = np.mean(Data,axis = 1)
	maximum = np.amax(Data,axis = 1)
	minimum = np.amin(Data,axis = 1)
	stdv = np.std(Data,axis = 1)
	freq_band = [0.1,4,8,12,30,70,180]

	binP = []
	timeDomain = []
	for i in range(channelNum):
		data = Data[i]
		if detec == 1:
			binP += list(pyeeg.bin_power(data,freq_band,sample_freq)[0])
		else:
			binP += list(pyeeg.bin_power(data,freq_band,sample_freq)[0]*(10**-5))
		timeDomain += list(pyeeg.hjorth(data))

		if detec == 1:
			timeDomain.append(pyeeg.hfd(data,4))

		timeDomain.append(pyeeg.pfd(data))

	'''
	fft = np.fft.fft(Data,2).flatten()
	print(len(Data[0]),len(fft))
	tmpf = [1,time]+list(mean)+list(maximum - minimum)+list(fft)+binP+timeDomain
	'''
	tmpf = [1,time]+list(normalization(mean))+list(normalization(maximum - minimum))+binP+timeDomain
	return np.nan_to_num(np.array(tmpf))

def detecData(observer):

	foldername = '..\\detection\\'+observer
	files = os.listdir(foldername)
	y = []
	X = []
	N = len(files) # number of data segments
	cn = 0
	for i in range(N): 
		fn = files[i]

		if 'test' in fn:
			continue

		mat = scipy.io.loadmat(foldername+'\\'+fn)
		k = list(mat.keys())
		emer = -1

		if len(k) == 7: # ictal
			y.append(-1)
			emer = mat[k[6]][0]
		else: # interictal
			y.append(1)
		cn,_ = mat[k[3]].shape
		X.append(extract(mat[k[3]],mat[k[4]],cn,emer))
		print('processing file %d'%(i+1))


	X = np.array(X)
	y = np.array(y)

	xn = 'Data\\D_' + observer + 'x.csv'
	yn = 'Data\\D_' + observer + 'y.csv'

	np.savetxt(xn, X, delimiter=",",header = head_generate(cn))
	np.savetxt(yn, y, delimiter=",")

def predictData(observer):
	
	foldername = '..\\prediction\\'+observer
	files = os.listdir(foldername)

	y = []
	X = []
	N = len(files) # number of data segments
	cn = 0
	for i in range(N):
		fn = files[i]

		if 'test' in fn:
			continue


		mat = scipy.io.loadmat(foldername+'\\'+fn)
		k = list(mat.keys())

		D = mat[k[3]][0][0][0]
		fs = mat[k[3]][0][0][2]
		s = mat[k[3]][0][0][4][0][0]
		cn,_ = D.shape
		w = len(D[0])//10
		
		Ds = D.T[w*2:w*3].T
		X.append(extract(Ds,fs,cn,s+0.2,0))

		Ds = D.T[w*5:w*6].T
		X.append(extract(Ds,fs,cn,s+0.5,0))

		Ds = D.T[w*7:w*8].T
		X.append(extract(Ds,fs,cn,s+0.7,0))

		if 'preictal' in k[3]: # preictal
			y += [-1] * 3
		else: # interictal
			y += [1] * 3
		

		print('processing %s file %d'%(observer, i+1))

	X = np.array(X)
	y = np.array(y)

	xn = 'Data\\P_' + observer + 'x.csv'
	yn = 'Data\\P_' + observer + 'y.csv'

	np.savetxt(xn, X, delimiter=",",header = head_generate(cn,0))
	np.savetxt(yn, y, delimiter=",")


def checkP():
	foldn = '..\\prediction\\Dog_1'

	files = os.listdir(foldn)

	# prediction
	# ['__header__', '__version__', '__globals__', 'interictal_segment_1']
	mat = scipy.io.loadmat(foldn+'\\'+files[0])
	k = list(mat.keys())
	print(k)
	print(mat[k[3]][0][0][0].shape)

def checkD():
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



# transform all the dataset
def transform():
	'''
	for i in range(1,5):
		observer = 'Dog_%d'%i
		detecData(observer)
	for i in range(1,9):
		observer = 'Patient_%d'%i
		detecData(observer)
	'''
	for i in range(2,6):
		observer = 'Dog_%d'%i
		predictData(observer)
	for i in range(1,3):
		observer = 'Patient_%d'%i
		predictData(observer)

transform()