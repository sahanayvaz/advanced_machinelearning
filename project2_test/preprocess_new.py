import numpy as np 
from scipy.ndimage.filters import gaussian_filter
from sklearn.feature_selection import VarianceThreshold

# gaussian filtering
def apply_gaussian(X, sigma, name):
	n_samples, n_features = X.shape
	X_3D = np.reshape(X, (-1, 176, 208, 176))
	X_filtered = np.reshape(X, (-1,176,208,176))
	for i in range(n_samples):
		print('gaussian filtering sample %d of ' %(i) + name)
		X_filtered[i,:,:,:] = gaussian_filter(X_3D[i,:,:,:], sigma)
	return np.reshape(X_filtered, (-1, 176*208*176))

# load data
print('loading data...')

X_train = np.load('./data/X_train.npy')
X_test = np.load('./data/X_test.npy')

print('done loading data')

# apply gaussian filter
print('applying gaussian filter...')

sigma = 1
X_gaussian = apply_gaussian(X_train, sigma, name='X_train')
Xt_gaussian = apply_gaussian(X_test, sigma, name='X_test')

print('done applying gaussian filter')

# remove 0 variance features
print('removing zero variance features...')
selector = VarianceThreshold()
selector.fit(X_gaussian)
X_gr_train = selector.transform(X_gaussian) 
X_gr_test = selector.transform(Xt_gaussian)
print('done removing')

print('saving to pickle...')
np.save('./data/X_gr_train.npy', X_gr_train)
np.save('./data/X_gr_test.npy', X_gr_test)