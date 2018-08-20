import numpy as np

# load data
print('loading data...')
X_real = np.load('./data/real_training.npy')
X_train = np.load('./data/train_data.npy')
X_test = np.load('./data/test_data.npy')

y_real = np.loadtxt('./data/real_labels.csv', dtype='int', delimiter=' ')
y_train = np.loadtxt('./data/y_train.csv', dtype='int', delimiter=' ')

n_real = X_real.shape[0]
n_train = X_train.shape[0]
n_test = X_test.shape[0]

X_real = X_real[:,0:9000]
X_train = X_train[:,0:9000]
X_test = X_test[:,0:9000]

n_real = X_real.shape[0]
n_train = X_train.shape[0]
n_test = X_test.shape[0]

print('matching')
match_count = 0
double_match_count = 0
match_list = []
y_index = np.zeros((n_test), dtype='int')

for i in range(n_real):
	if (i % 100) == 0:
		print('we are at %d' %(i))
	for j in range(n_test):
		diff = X_real[i,:] - X_test[j,:]
		if np.mean(diff) == 0.0:
			#print('we get match for %d' %(j))
			if j not in match_list:
				match_count += 1
				match_list.append(j)
			else:
				print('double match')
				double_match_count += 1
			y_index[j] = i

y_labels = np.zeros((n_test,4))
for i in range(n_test):
	y_labels[i,:] = y_real[y_index[i],:]

print(len(match_list))
print(double_match_count)
np.savetxt('./data/y_test.csv', y_labels, fmt='%d')
