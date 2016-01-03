import numpy as np
from keras.utils import np_utils, generic_utils

def load_data():
	print("Loading data...")

	X_train = np.load('./tratz_training_features.npy')
	y_train = np.load('./tratz_training_labels.npy')
	print(len(X_train), 'train instances')

	X_test  = np.load('./tratz_test_features.npy')
	print(len(X_test), 'test instances')

	nb_features = X_train.shape[1]
	nb_classes = np.max(y_train)+1
	print(nb_classes, 'classes')
	print(X_train)

	X_train = X_train
	X_test = X_test

	Y_train = np_utils.to_categorical(y_train, nb_classes)



load_data()