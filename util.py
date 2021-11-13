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

foldn = '..\\prediction\\Dog_1'

files = os.listdir(foldn)

# ['__header__', '__version__', '__globals__', 'interictal_segment_1']
mat = scipy.io.loadmat(foldn+'\\'+files[0])
k = list(mat.keys())
print(k)
print(mat[k[3]][0][0][0].shape)

foldn = '..\\detection\\Dog_1'
files = os.listdir(foldn)

# ['__header__', '__version__', '__globals__', 'data', 'freq', 'channels', 'latency'] -- ictal
# OR
# ['__header__', '__version__', '__globals__', 'data', 'freq', 'channels'] -- interictal
mat = scipy.io.loadmat(foldn+'\\'+files[2000])
k = list(mat.keys()) # 
print(k)
print(mat[k[3]].shape)

