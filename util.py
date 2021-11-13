import scipy.io
import os
import numpy as np

def getData(fn,fold = 'detection'):
	name = fold + '\\' + fn

	#dict_keys(['__header__', '__version__', '__globals__', 'X', 'lambdas', 'y'])
	mat = scipy.io.loadmat(fn)

	Dx = mat['X']
	Dy = mat['y']

	state = np.random.get_state()
	np.random.shuffle(Dx)
	np.random.set_state(state)
	np.random.shuffle(Dy)

	# N = 351
	Dy = np.ravel(Dy)
	#print(y.shape)


	return Dx, Dy

foldn = '..\\detection\\Dog_1'
foldn = '..\\prediction\\Dog_1'

files = os.listdir(foldn)

# dict_keys(['__header__', '__version__', '__globals__', 'data', 'freq', 'channels', 'latency'])
mat = scipy.io.loadmat(foldn+'\\'+files[0])
k = list(mat.keys())
print(k)
print(mat['interictal_segment_1'][0][0][0].shape)