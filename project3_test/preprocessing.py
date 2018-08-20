import numpy as np
#import keras.utils as np_utils
from scipy.signal import butter, lfilter, filtfilt

## apply butterworthbandpass
def butter_bandpass(lowcut, highcut, fs, order):
   	nyq = 0.5 * fs
   	low = lowcut / nyq
   	high = highcut / nyq
   	b, a = butter(order, [low,high], btype='bandpass')
   	return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
   	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = filtfilt(b, a, data)
	return y

def preprocess(X):
	n_samples, n_features = X.shape
	X_preprocessed = np.zeros((n_samples, n_features))
	for i in range(n_samples):
		x = X[i,:]
		x = x - np.mean(x)
		x = x / np.std(x)
		X_preprocessed[i,:] = x
	return X_preprocessed

# load data
X_train = np.load('./data/train_data.npy')
X_test = np.load('./data/test_data.npy')

X_train = X_train[:,0:9000]
X_test = X_test[:,0:9000]

# wide butterworth filter
highcut = 45
lowcut = 5
fs = 300.0
order = 10
n_samples, n_features = X_train.shape
X_bw = np.zeros((n_samples, n_features))
n_samp, n_feat = X_test.shape
Xt_bw = np.zeros((n_samp, n_feat))

for i in range(n_samples):
	X_bw[i, :] = (butter_bandpass_filter(X_train[i,:], lowcut, highcut, fs, order))

for i in range(n_samp):
	Xt_bw[i, :] = (butter_bandpass_filter(X_test[i,:], lowcut, highcut, fs, order))

#X_train = preprocess(X_train)

X_bw = preprocess(X_bw)
Xt_bw = preprocess(Xt_bw)

print('saving')
np.save('./data/X_bw.npy', X_bw)
np.save('./data/Xt_bw.npy', Xt_bw)