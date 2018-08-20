import numpy as np 
from sklearn.metrics import f1_score, accuracy_score

def get_distro(y):
	# get the distribution of labels
	n_samples = y.shape[0]
	zeros = 0
	ones = 0
	twos = 0
	threes = 0
	for i in range(n_samples):
		if y[i] == 0:
			zeros += 1.0
		elif y[i] == 1:
			ones += 1.0
		elif y[i] == 2:
			twos += 1.0
		else:
			threes += 1.0

	dis_zeros = zeros / n_samples
	dis_ones = ones / n_samples
	dis_twos = twos / n_samples
	dis_threes = threes /n_samples
	return [dis_zeros, dis_ones, dis_twos, dis_threes]

# load data
y_train = np.loadtxt('./data/train_labels.csv')

# this is our new prediction
y_pred = np.loadtxt('./data/y_test_conv.csv', delimiter=' ')
y_pred = np.argmax(y_pred, axis=1)
print(y_pred.shape)

# getting submission to check
submission = np.loadtxt('./data/submission1.csv', delimiter=',')
idx = submission[:,0]
submission = submission[:,1]
sub = np.hstack((idx[:, None], y_pred[:, None]))
print(sub.shape)
np.savetxt('./data/submission_cnn.csv', sub, fmt='%d', delimiter=',')

test = np.loadtxt('./data/submission_cnn.csv', dtype='int', delimiter=',')
test = test[:,1]

# correcting y_test
y_test = np.loadtxt('./data/y_test.csv', dtype='int', delimiter=' ')
y_temp = y_test[:,0:1]
print(y_temp.shape)
y_temp2 = y_test[:,1:4]
print(y_temp2.shape)
y_test = np.hstack((y_temp2, y_temp))
y_test = np.argmax(y_test, axis=1)

f1 = f1_score(y_pred, y_test, average='micro')
acc = accuracy_score(y_pred, y_test)
print(f1)
print(acc)

f1 = f1_score(test, y_test, average='micro')
acc = accuracy_score(test, y_test)
print(f1)
print(acc)

f1 = f1_score(submission, y_test, average='micro')
acc = accuracy_score(submission, y_test)
print(f1)
print(acc)

print('distribution of y_train: ')
print(get_distro(y_train))
print('distribution of submission: ')
print(get_distro(submission))
print('distribution of y_test: ')
print(get_distro(y_test))
print('distribution of y_pred: ')
print(get_distro(y_pred))