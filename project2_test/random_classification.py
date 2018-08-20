import numpy as np 
from scipy.stats import spearmanr as sp 
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.random import sample_without_replacement
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# evaluation metric 
def sp_eval(y_pred, y_true):
	n, m = y_pred.shape
	total = 0
	for i in range(n):
		rho, pval = sp(y_pred[i,:], y_true[i,:])
		total += rho
	return total/n

# load data
print('loading data ...')
#X_train = np.load('./data/X_train.npy')
#X_test = np.load('./data/X_test.npy')

X_train = np.load('./data/X_gr_train.npy')
X_test = np.load('./data/X_gr_test.npy')

print(X_train.shape, X_test.shape)

y_train_probs_file = './data/train_labels_new.csv'
y_train_labels_file = './data/y_labels.csv'
y_train_probs = np.loadtxt(y_train_probs_file)
y_train_labels = np.loadtxt(y_train_labels_file)
print('done loading')

n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=False)

n_samples, n_features = X_train.shape
n_components = 10000
print('ready to train')

means = np.zeros((100,1))

for i in range(100):
	mean = 0
	print('random sampling on iteration %d' %(i))
	idx = sample_without_replacement(n_features, n_components)
	X_random = X_train[:,idx]
	# logistic regression
	for train_index, test_index in skf.split(X_random, y_train_labels):
		# get train and test sets from X_train
		Z_train, Z_test = X_random[train_index,:], X_random[test_index,:]
		# get train and test labels from y_train_labels
		w_train_labels, w_test_labels = y_train_labels[train_index], y_train_labels[test_index]
		# get train and test probs from y_train_probs
		w_train_probs, w_test_probs = y_train_probs[train_index,:], y_train_probs[test_index,:]
		
		#logistic = LogisticRegression(C=0.1, solver='newton-cg', max_iter=200, multi_class='multinomial')
		#logistic.fit(Z_train, w_train_labels)
		#w_pred_probs = logistic.predict_proba(Z_test)

		#rfc = AdaBoostClassifier(n_estimators=1000)
		#rfc.fit(Z_train, w_train_labels)
		#w_pred_probs = rfc.predict_proba(Z_test)
		
		mean_rho = sp_eval(w_pred_probs, w_test_probs)
		mean += mean_rho
	means[i] = mean/n_splits
	print('mean eval of k-fold training: %0.4f' % (mean/n_splits))
	print('done k-fold training on %d th sample' %(i))
print(np.max(means))