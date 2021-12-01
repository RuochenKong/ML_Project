import numpy as np
from cssvm import Loss
from cssvm import LossGrad
import matplotlib.pyplot as plt
from sklearn import svm

def generate(mean, cova, num):
	sigma = np.eye(2)
	miu = np.zeros((2,))

	Z = np.random.multivariate_normal(miu, sigma, num)

	cov = np.array(cova)
	u,s,vt = np.linalg.svd(cov, full_matrices=True)

	sigma[0][0] = s[0]
	sigma[1][1] = s[1]


	miu = np.array(mean)
	sigma = sigma **(1/2)

	X = u.dot(sigma.dot(Z.T))
	X = X + miu

	return X

def Mysvm(Data,y,C,r1,r2,eta):

	N,p = Data.shape
	w = np.random.uniform(-1,1,p)

	err = 1
	lO = 0
	MaxIter = 10**3
	k = 0
	while (err > 10**(-4) and k < MaxIter):

		for i in range(N):
			w = w - eta*LossGrad(w,Data[i],y[i],C,r1,r2)
		l = Loss(w,Data,y,C,r1,r2)
		err = abs(l-lO)
		lO = l
		k += 1

	return w


m = [[2],[2]]
c = [[2,-1],[-1,1]]
X1 = generate(m,c,3000).T
m = [[0],[0]]
c = [[1,0.5],[0.5,1]]
X2 = generate(m,c,200).T
X = np.concatenate((X1,X2),axis = 0)
Dx  = np.zeros((3200,1))
Dx = np.concatenate((Dx,X),axis = 1)


y = [1]*3000 + [-1]*200

y = np.array(y)

C = 2**(-7)
r1 = 2**8
r2 = 1
w = Mysvm(Dx,y,C,r1,r2,10**(-4))
#w = Mysvm(Dx,y,C,0.5,0.5,10**(-4))
yh = np.sign(Dx.dot(w))

D1 = X[yh > 0]
D2 = X[yh < 0]

plt.figure(1)
plt.scatter(D1.T[0],D1.T[1])
plt.scatter(D2.T[0],D2.T[1])

cor = 0

for i in range(3200):
	if yh[i] == y[i] and y[i] < 0:
		cor += 1
print(cor/200)

print(len(yh[yh != y])/3200)

plt.show()

'''
clf = svm.SVC()
clf.fit(X, y)
yh = clf.predict(X)

print()
cor = 0

for i in range(3200):
	if yh[i] == y[i] and y[i] < 0:
		cor += 1
print(cor/200)

print(len(yh[yh != y])/3200)

clf = svm.SVC(class_weight='balanced')
clf.fit(X, y)
yh = clf.predict(X)

print()
cor = 0

for i in range(3200):
	if yh[i] == y[i] and y[i] < 0:
		cor += 1
print(cor/200)

print(len(yh[yh != y])/3200)
'''