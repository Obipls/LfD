from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, MaxoutDense
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from collections import Counter

max_words = 500
batch_size = 32
nb_epoch = 5

def NNclassify(X_train,X_test,y_train,y_test,inputtype):
	print('Loading data...')
	print(len(X_train), 'train instances')
	print(len(X_test), 'test instances')

	nb_classes = np.max(y_train)+1
	print(nb_classes, 'classes')

	print('Vectorizing sequence data...')
	tokenizer = Tokenizer(nb_words=max_words)
	X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
	X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
	print('X_train shape:', X_train.shape)
	print('X_test shape:', X_test.shape)

	print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)
	print('Y_train shape:', Y_train.shape)
	print('Y_test shape:', Y_test.shape)

	print('Building model...')
	model = Sequential()
	model.add(MaxoutDense(1, input_shape=(max_words,)))
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.1))
	model.add(MaxoutDense(nb_classes,max_words))
	model.add(Activation('sigmoid'))

	model.compile(loss='categorical_crossentropy', optimizer='adam',class_mode=inputtype)
	history = model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1, show_accuracy=True, validation_split=0.1)
	score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1, show_accuracy=True)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])

	prediction=model.predict(X_test, batch_size=batch_size, verbose=1)
	pred_classes = np.argmax(prediction, axis=1)
	print(Counter(pred_classes))

	return pred_classes