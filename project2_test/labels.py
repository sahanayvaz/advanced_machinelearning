import numpy as np 

print('loading y_label_probs')
y_train_probs = np.loadtxt('./data/train_labels_new.csv')
print(y_train_probs.shape)
print('done loading y_label_probs')

# we need to convert max probabilities to labels
# well this was way simpler than i thought
print('converting max probs to labels')
y_train_labels = np.argmax(y_train_probs, axis=1)
print(y_train_labels.shape)
print('done converting')

np.savetxt('./data/y_labels.csv', y_train_labels, fmt='%d')