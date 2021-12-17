import numpy as np
from numpy import genfromtxt
from sklearn import svm
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

def loadData(ob,detec = 1):
	xn = ''
	yn = ''
	if detec == 1:
		xn = 'Data\\D_'+ob+'x.csv'
		yn = 'Data\\D_'+ob+'y.csv'
	else:
		xn = 'Data\\P_'+ob+'x.csv'
		yn = 'Data\\P_'+ob+'y.csv'

	return genfromtxt(xn, delimiter=','),genfromtxt(yn, delimiter=',')


observer = 'Dog_5'
X,yprim = loadData(observer,0)


y = np.zeros(yprim.shape)

y[yprim<0] += 1


NumIc = len(y[y==1])
NumInter = len(y) - NumIc

state = np.random.get_state()
np.random.shuffle(X)
np.random.set_state(state)
np.random.shuffle(y)

N,p = X.shape
print(N)
# seperate data for cross-validation
Vx = []
Vy = []
n = N//5

for i in range(4):
	Xi = X[i*n:(i+1)*n][:]
	yi = y[i*n:(i+1)*n]
	Vx.append(Xi)
	Vy.append(yi)

Vx.append(X[4*n:])
Vy.append(y[4*n:])

AUC = 0

clf = svm.SVC()
for testi in range(5):

	Vxtest = Vx[testi]
	Vytest = Vy[testi]

	I = [0,1,2,3,4]
			
	I.remove(testi)

	Vyprob = np.zeros(Vytest.shape)

	for i in range(4):

		Vxvalid = Vx[I[i]]
		Vyvalid = Vy[I[i]]
		trainI = []
		for tmp in I:
			trainI.append(tmp)
		trainI.remove(I[i])


		Vxtrain = np.concatenate((Vx[trainI[0]],Vx[trainI[1]],Vx[trainI[2]]),axis = 0)
		Vytrain = np.concatenate((Vy[trainI[0]],Vy[trainI[1]],Vy[trainI[2]]))

		clf.fit(Vxtrain,Vytrain)
				
		Vyprob += clf.predict(Vxtest)

	Vyprob /= 4
	auc = metrics.roc_auc_score(Vytest, Vyprob)
	AUC += auc

AUC /= 5

print(AUC)