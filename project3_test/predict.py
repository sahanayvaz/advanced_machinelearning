import numpy as np
from keras.models import load_model
# keras imports
import keras
from keras.models import Model
from keras.layers import Input, Conv1D, Dropout, Flatten, Dense, MaxPooling1D, Activation, GlobalAveragePooling1D, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import Callback,warnings
import keras.utils as np_utils

import tensorflow as tf 

def cnn_net(window_size):
	# parameters
	input_feat = 1
	output_feat = 4

	convfilt = 128 		# number of neurons
	convstr = 1 		# no idea what it does 
	ksize = 5			# kernel width	
	dropout = 0.20		# dropout probability

	poolsize = 2
	poolstr = 2

	# input layer
	input1 = Input(shape=(window_size, input_feat), name='input1')

	# first convolution layer conv(relu)->maxpooling->dropout
	x = Conv1D(filters=convfilt,
				   kernel_size=ksize,
				   padding='same',
				   strides=convstr,
				   kernel_initializer='he_normal')(input1)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling1D(pool_size=poolsize,
						strides=poolstr)(x)
	#x = Dropout(dropout)(x)

	# other 6 convolution layers conv(relu)->maxpooling->dropout
	for i in range(6):
		x = Conv1D(filters=convfilt,
					   kernel_size=ksize,
					   padding='same',
					   strides=convstr,
					   kernel_initializer='he_normal')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = MaxPooling1D(pool_size=poolsize,
							strides=poolstr)(x)
		#x = Dropout(dropout)(x)

	x = GlobalAveragePooling1D()(x)

	# not sure about this step
	#x = Flatten()(x)

	# dense layers
	x = Dense(256, activation='relu')(x)
	x = Dropout(dropout)(x)
	x = Dense(128, activation='relu')(x)
	x = Dropout(dropout)(x)
	x = Dense(64, activation='relu')(x)
	x = Dropout(dropout)(x)

	# output
	out = Dense(output_feat, activation='softmax')(x)
	model = Model(inputs=input1, outputs=out)

	model.compile(optimizer='adam',
				  loss='categorical_crossentropy',
				  metrics=['accuracy'])
	return model

# Load data
X_test = np.load('./data/Xt_bw.npy')

# Parameters
FS = 300
maxlen = 30*FS

'''
# Preprocessing data
n_samples, n_features = X_test.shape
data = np.zeros((n_samples, n_features))
for i in range(n_samples):
    x = X_test[i,:]
    x = x - np.mean(x)
    x = x / np.std(x)
    data[i,:] = x
data = np.expand_dims(data, axis=2)
'''
model = load_model('128_conv_7_dense_3.h5')

#model1 = cnn_net(maxlen)
#model1.load_weights('weights.h5')

data = np.expand_dims(X_test, axis=2)

print("Applying model ..")    
prob = model.predict(data)
np.savetxt('./data/y_test_conv.csv', prob, fmt='%0.2f')
