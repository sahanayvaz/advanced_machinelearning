import numpy as np
from sklearn.model_selection import train_test_split

# keras imports
import keras
from keras.models import Model
from keras.layers import Input, Conv1D, Dropout, Flatten, Dense, MaxPooling1D, Activation, GlobalAveragePooling1D, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import Callback,warnings
import keras.utils as np_utils

class AdvancedLearnignRateScheduler(Callback):    
    def __init__(self, monitor='val_loss', patience=0,verbose=0, mode='auto', decayRatio=0.1):
        super(Callback, self).__init__() 
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.decayRatio = decayRatio
 
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('Mode %s is unknown, '
                          'fallback to auto mode.'
                          % (self.mode), RuntimeWarning)
            mode = 'auto'
 
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
 
    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        current_lr = K.get_value(self.model.optimizer.lr)
        print("\nLearning rate:", current_lr)
        if current is None:
            warnings.warn('AdvancedLearnignRateScheduler'
                          ' requires %s available!' %
                          (self.monitor), RuntimeWarning)
 
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('\nEpoch %05d: reducing learning rate' % (epoch))
                    assert hasattr(self.model.optimizer, 'lr'), \
                        'Optimizer must have a "lr" attribute.'
                    current_lr = K.get_value(self.model.optimizer.lr)
                    new_lr = current_lr * self.decayRatio
                    K.set_value(self.model.optimizer.lr, new_lr)
                    self.wait = 0 
            self.wait += 1

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

#####################
### MAIN FUNCTION ###
#####################

X_train = np.load('./data/X_bw.npy')
X_test = np.load('./data/Xt_bw.npy')

y_train = np.loadtxt('./data/train_labels.csv', dtype='int')

WINDOW_SIZE = 300*30
X_train = X_train[:,0:WINDOW_SIZE]
X_test = X_test[:,0:WINDOW_SIZE]

# add dimensions for training
#X_tr, y_tr, X_val, y_val = train_test_split(X_train, y_labels, test_size=0.20)

X_tr = np.expand_dims(X_train, axis=2)
y_tr = np_utils.to_categorical(y_train, num_classes=4)
X_test = np.expand_dims(X_test, axis=2)

#X_val = np.expand_dims(X_val, axis=2)
#y_val = np_utils.to_categorical(y_val, num_classes=4)

# initiate model
model = cnn_net(WINDOW_SIZE)
epochs = 25
batch = 64
callbacks = [
   	EarlyStopping(monitor='val_loss', patience=3, verbose=1),
   	AdvancedLearnignRateScheduler(monitor='val_loss', patience=1, verbose=1, mode='auto', decayRatio=0.1),            
 	]
print(model.count_params())
model.fit(X_tr, y_tr, validation_split=0.1, epochs=epochs, batch_size=batch, verbose=1, callbacks=callbacks)
print('applying model')
y_pred = model.predict(X_test)
np.savetxt('./data/y_train_conv.csv', y_pred, fmt='%d')
