import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def prepare(X, index):
	X = X[index, :]
	#idx = np.nonzero(X)
	#X = X[idx]
	X = pd.DataFrame(X)
	return X


X_train = np.load('./data/X_train.npy')
X_train = X_train[:,0:9000]

y_train = np.loadtxt('./data/train_labels.csv')
'''
X_test = np.load('./data/test_data.npy')
print(X_test.shape)
#np.savetxt('./data/feature_distro.csv', non_zero_features(X_train, y_train), fmt='%0.2f', delimiter=',')
'''

X_bw = np.load('./data/X_bw.npy')
print(X_bw[0,:])
print(X_bw.shape)
one = 3
#two = 300
#three = 400

X_one = prepare(X_train, one)
X_two = prepare(X_bw, one)
print(y_train[one])
#X_two = prepare(X_train, two)
#X_three =prepare(X_train, three)

ax = X_one.plot()
X_two.plot(ax=ax)
#X_three.plot()
plt.show()
