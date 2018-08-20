import numpy as np 
from scipy import stats

# average voxels
def calculate_voxels(X, voxel_size):
    #dimension of the data is 176 208 176
    xShape = 176/voxel_size 
    yShape = 208/voxel_size 
    zShape = 176/voxel_size
    
    n = X.shape[0]
    meanFeatures = np.zeros((n,  xShape*yShape*zShape))
    maxFeatures = np.zeros((n,  xShape*yShape*zShape))
    minFeatures = np.zeros((n,  xShape*yShape*zShape))
    medianFeatures = np.zeros((n,  xShape*yShape*zShape))    
    modeFeatures = np.zeros((n,  xShape*yShape*zShape))    
    stdFeatures = np.zeros((n,  xShape*yShape*zShape))

    # reshape
    X_reshape = X.reshape(-1, 176, 208, 176)
    # calculate mean of voxel pixels
    # remember to change that
    for i in xrange(n):
        print('calculating mean on sample %d' %i)
        for x in xrange(xShape):
            for y in xrange(yShape):
                for z in xrange(zShape):
                    voxel = X_reshape[i, (voxel_size*x):(voxel_size*x+voxel_size), (voxel_size*y):(voxel_size*y+voxel_size),(voxel_size*z):(voxel_size*z+voxel_size)]
                    voxel = voxel.ravel()
                    meanFeatures[i, yShape*zShape*x+xShape*y+z] = np.mean(voxel)
                    maxFeatures[i, yShape*zShape*x+xShape*y+z] = np.amax(voxel)
                    minFeatures[i, yShape*zShape*x+xShape*y+z] = np.amin(voxel)
                    medianFeatures[i, yShape*zShape*x+xShape*y+z] = np.median(voxel)
                    mode = stats.mode(voxel, axis=None)
                    modeFeatures[i, yShape*zShape*x+xShape*y+z] = mode.mode[0]
                    stdFeatures[i, yShape*zShape*x+xShape*y+z] = np.std(voxel)
    features = np.concatenate((meanFeatures,maxFeatures,minFeatures, medianFeatures, modeFeatures, stdFeatures), axis=1)
    return features

# normalize the data
def normalize(X, X_test):
	X_mean = np.mean(X, axis=0)
	X_std = np.std(X, axis=0)
	# remove zero variance features
	idx = np.nonzero(X_std)
	X = X[:,idx[0]]
	X_test = X_test[:,idx[0]]
	X_mean = X_mean[idx[0]]
	X_std = X_std[idx[0]]
	X = (X-X_mean)/X_std
	X_test = (X_test-X_mean)/X_std
	return X, X_test

#load data
print('loading data ...')
X_train = np.load("./data/X_train.npy")
X_test = np.load("./data/X_test.npy")
print('done loading')

# get mean of voxels
print('processing voxels ...')
voxel_size = 8
X_train = calculate_voxels(X_train, voxel_size)
X_test = calculate_voxels(X_test, voxel_size)
print('done processing voxels')

# preprocess to zero-mean, unit variance
#print('beginning normalizing ...')
#X_train, X_test = normalize(X_train, X_test)
#print(X_train.shape)
#print(X_test.shape)
#print('done normalizing')

np.savetxt('./data/X_v8_preprocessed_no.csv', X_train, delimiter=',')
np.savetxt('./data/Xt_v8_preprocessed_no.csv', X_test, delimiter=',')