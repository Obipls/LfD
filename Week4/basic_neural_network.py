#!/usr/bin/env python

import numpy as np
np.random.seed(1337)  # for reproducibility
import argparse

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, MaxoutDense
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils, generic_utils
from keras.optimizers import *

class KerasNN(object):

    def __init__(self, n_units, n_epochs, batch_size, name):

        # Parameters
        self.batch_size          = batch_size
        self.n_epochs            = n_epochs
        self.layer1_hidden_units = n_units
        self.name                = name

        # Run the neural net
        self.load_data()
        self.build_model()
        self.train_model()
        self.test_model()

    def load_data(self):
        print("Loading data...")

        X_train = np.load('./tratz_training_features.npy')
        y_train = np.load('./tratz_training_labels.npy')
        print(len(X_train), 'train instances')

        X_test  = np.load('./tratz_test_features.npy')
        print(len(X_test), 'test instances')

        self.nb_features = X_train.shape[1]
        print(X_train.shape)
        print("dinges", self.nb_features)
        self.nb_classes = np.max(y_train)+1
        print(self.nb_classes, 'classes')

        self.X_train = X_train
        self.X_test = X_test

        self.Y_train = np_utils.to_categorical(y_train, self.nb_classes)

    def build_model(self):
        print("Building model...")
        self.model = Sequential() #Stacks all layers linearly

        #Inputs the features into the layers of the system
        # Changed from Dense to MaxoutDense, beacuse it is an optimized extension of the default Dense function. Achieves significant better (+5%)
        self.model.add(MaxoutDense(self.layer1_hidden_units, input_dim=self.nb_features))
        self.model.add(Activation('sigmoid'))

        #inputs the classes into the layers of the system
        # Different activations seem not to work. (wanted relu)
        self.model.add(MaxoutDense(self.nb_classes, input_dim=self.layer1_hidden_units))
        self.model.add(Activation('softmax'))

        # Compile the complete model, with the added classes and features
        # Optimizer Adam outperforms all others and speeds up the system compared to SGD. Adds up to 5% to the score
        self.model.compile(loss='categorical_crossentropy', optimizer='Adam')

        # Dropout randomly selects neurons to be set to zero, thus preventing the system from overfitting. No significant difference.
        self.model.add(Dropout(0.1))

    def train_model(self):
        print("Training model...")
        n_instances = self.X_train.shape[0]
        dev_split   = int(n_instances * 0.8)

        X_train = self.X_train[:dev_split]
        Y_train = self.Y_train[:dev_split]
        X_val   = self.X_train[dev_split:]
        Y_val   = self.Y_train[dev_split:]

        history = self.model.fit(X_train, Y_train, nb_epoch=self.n_epochs, batch_size=self.batch_size, shuffle='batch', verbose=1, show_accuracy=True, validation_data=(X_val, Y_val), )

    def test_model(self):
        outputs = self.model.predict(self.X_test, batch_size=self.batch_size)
        pred_classes = np.argmax(outputs, axis=1)
        print(pred_classes)

        np.save(self.name, pred_classes)
# Several 'classic' parameters that change the performance significantly (+50% in the best case)
parser = argparse.ArgumentParser(description='KerasNN parameters')
parser.add_argument('--units', metavar='xx', type=int, default=500, help='units')
parser.add_argument('--epochs', metavar='xx', type=int, default=20, help='epochs')
parser.add_argument('--bsize', metavar='xx', type=int, default=50, help='batch size')
parser.add_argument('--name', type=str, default='Olivier_3', help='output file name')
args = parser.parse_args()
if __name__ == '__main__':
    KerasNN(n_units=args.units,
            n_epochs=args.epochs,
            batch_size=args.bsize,
            name=args.name)