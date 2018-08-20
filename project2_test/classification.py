import numpy as np 
from scipy.stats import spearmanr as sp 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

def sp_eval(y_pred, y_true):
	n, m = y_pred.shape
	total = 0
	for i in range(n):
		rho, pval = sp(y_pred[i,:], y_true[i,:])
		total += rho
	return total/n

# load data
print('loading data')
# no normalization for ensemble learning
X_train_file = './data/X_v8_preprocessed.csv'
X_test_file = './data/Xt_v8_preprocessed.csv'
y_train_probs_file = './data/train_labels_new.csv'
y_train_labels_file = './data/y_labels.csv'

X_train = np.loadtxt(X_train_file, delimiter=',')
X_test = np.loadtxt(X_test_file, delimiter=',')
y_train_probs = np.loadtxt(y_train_probs_file)
y_train_labels = np.loadtxt(y_train_labels_file)
print('done loading')

'''
print('beginning k-fold training')
# kfold: shuffle=False, random_state=None
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
mean = 0
# logistic regression
for train_index, test_index in skf.split(X_train, y_train_labels):
	# get train and test sets from X_train
	Z_train, Z_test = X_train[train_index,:], X_train[test_index,:]
	# get train and test labels from y_train_labels
	w_train_labels, w_test_labels = y_train_labels[train_index], y_train_labels[test_index]
	# get train and test probs from y_train_probs
	w_train_probs, w_test_probs = y_train_probs[train_index,:], y_train_probs[test_index,:]
	#logistic = LogisticRegression(C=0.01, solver='newton-cg', multi_class='multinomial')
	#logistic.fit(Z_train, w_train_labels)
	#w_pred_probs = logistic.predict_proba(Z_test)
	rfc = RandomForestClassifier(n_estimators=5000)
	rfc.fit(Z_train, w_train_labels)
	w_pred_probs = rfc.predict_proba(Z_test)
	mean_rho = sp_eval(w_pred_probs, w_test_probs)
	mean += mean_rho
	print(mean_rho)
print('mean eval of k-fold training: %0.2f' % (mean/n_splits))
print('done k-fold training')


'''
print('training the model')

#logistic = LogisticRegression(C=0.01, solver='newton-cg', multi_class='multinomial')
#logistic.fit(X_train, y_train_labels)
rfc = RandomForestClassifier(n_estimators=5000)
rfc.fit(X_train, y_train_labels)
print('done training')

print('predicting probabilities')
y_test_probs = rfc.predict_proba(X_test)
print('done predicting')

print('saving .csv')
n = y_test_probs.shape[0]
submission_test = np.zeros((n, 5))

for i in range(n):
	submission_test[i,0] = i+1
submission_test[:,1::] = y_test_probs

np.savetxt('submission_test_4.csv', submission_test, fmt='%d %0.5f %0.5f %0.5f %0.5f')
print('done saving')