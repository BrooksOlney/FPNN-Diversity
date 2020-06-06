from tensorflow import keras
from tensorflow import Graph
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout, AveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from copy import deepcopy
import multiprocessing
import time as t
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import struct
from dataset.dataset import dataset
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
	
# tf.keras.backend.set_epsilon(1e-4)
# tf.keras.backend.set_floatx('float16')

class top_model:
    def __init__(self):
        # declare sequential model, load MNIST dataset
        self.model = Sequential()
        # self.dataset = dataset()

        # create simple keras model 
        self.model.add(Conv2D(6, (3, 3), input_shape=(28, 28, 1), activation='relu'))
        # self.model.add(Dropout(0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(16, (3, 3), activation='relu'))
        # self.model.add(Dropout(0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(120, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(84, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        self.deltas = None
        self.orig_weights = None

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def train_model(self, dataset):
        self.model.fit(dataset.train_X, dataset.train_Y_one_hot, batch_size=1024, epochs=10, verbose=0)
        self.orig_weights = self.model.get_weights()

    def poisoned_retrain(self, dataset, num_samples, num1, num2, epochs):
        if self.orig_weights is not None:
            del self.orig_weights

        self.orig_weights = deepcopy(self.model.get_weights())
        dataset.label_flip(num_samples, num1, num2)
        # self.model.fit(dataset.poisoned_X, dataset.poisoned_Y_one_hot, batch_size=1024, epochs=epochs, verbose=0, validation_data=(dataset.pvalid_X, dataset.pvalid_Y_one_hot))
        self.model.fit(dataset.poisoned_X, dataset.poisoned_Y_one_hot, batch_size=1024, epochs=epochs, verbose=0)
        self.create_update()

    def test_model(self, dataset):
        pred_y = self.model.predict_on_batch(dataset.test_X)
        test_acc = np.mean(np.argmax(pred_y, axis=1) == dataset.test_Y)
        
        return test_acc

    def plot_cf(self, dataset):
        fig = plt.figure(figsize=(5, 5))

        y_pred = self.model.predict(dataset.test_X)
        Y_pred = np.argmax(y_pred, 1)
        Y_test = np.argmax(dataset.test_Y_one_hot, 1)
            
        mat = confusion_matrix(Y_test, Y_pred)
        for i in range(10):
            mat[i,i] = 0


        sns.heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues, fmt='g')
        plt.xlabel('Predicted Values')
        plt.ylabel('True Values')
        plt.show()

    def plot_diff_cf(self, dataset):
        fig = plt.figure(figsize=(5, 5))

        Y_pred_poisoned = np.argmax(self.model.predict(dataset.test_X), 1)
        
        self.set_weights(self.orig_weights)
        Y_pred_orig = np.argmax(self.model.predict(dataset.test_X), 1)

        self.set_weights(xor_weights(self.orig_weights, self.deltas))

        Y_test = np.argmax(dataset.test_Y_one_hot, 1)
            
        mat1 = confusion_matrix(Y_test, Y_pred_orig)
        mat2 = confusion_matrix(Y_test, Y_pred_poisoned)
        for i in range(10):
            mat1[i,i] = 0
            mat2[i,i] = 0

        resultingMat = mat2 - mat1

        sns.heatmap(resultingMat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues, fmt='g')
        plt.xlabel('Predicted Values')
        plt.ylabel('True Values')
        plt.show()

    def diversify_weights(self, percentage):
        # startTime = t.time()
        # logfile = open("logfile.txt", "a")
        self.orig_weights = deepcopy(self.model.get_weights())

        total_hamming = 0
        count = 0
        all_weights = self.model.get_weights()
        result = []
        for i in range(len(all_weights)):
        # for layer_weights in all_weights:
            layer_weights = all_weights[i]

            # # # skip the bias terms
            # if i % 2 == 1:
            #     # continuer
            #     result.append(layer_weights)
            #     continue

            # iterate through each layer and shift the weights, compute avg hamming distance
            new_weights = shift(layer_weights, percentage)
            result.append(new_weights)
            total_hamming += hamming(layer_weights, new_weights)
            count += layer_weights.size

        self.model.set_weights(result)
        total_hamming /= count
        avg_hamming = total_hamming

        # logfile.write("Diversify_weights ET: " + str(t.time() - startTime) + "s\n")
        # logfile.write("Count of weights: " + str(count) + "\n")
        # logfile.close()
        return avg_hamming

    def compute_probabilities(self):
        # compute probability of each bit flipping based on frequency and total # of weights

        totals = np.zeros((32))
        total_weights = 0

        for orig_weights, cur_weights in zip(self.orig_weights, self.model.get_weights()):
            num_weights = orig_weights.size
            total_weights += num_weights
            
            xor = (orig_weights.view('i') ^ cur_weights.view('i'))
            unpacked = np.unpackbits(xor.view('uint8'), bitorder='little')
            binned = np.array_split(unpacked, num_weights)

            totals += np.sum(binned, axis=0)

        return totals / total_weights

    def create_update(self, filename=None):
        if self.deltas is not None:
            del self.deltas

        self.deltas = deepcopy(xor_weights(self.model.get_weights(), self.orig_weights))
        
        if filename is not None:
            self.model.set_weights(self.deltas)
            self.model.save_weights(filename)
            self.model.set_weights(self.orig_weights)
        
    def update_network(self, update, filename=None):
        if self.orig_weights is not None:
            del self.orig_weights

        self.orig_weights = deepcopy(self.model.get_weights())

        if filename is not None:
            self.model.load_weights(filename)
            self.deltas = deepcopy(self.model.get_weights())
            self.model.set_weights(xor_weights(self.model.get_weights(), self.orig_weights))
        else: 
            self.model.set_weights(xor_weights(self.model.get_weights(), update))

    def reset_network(self):
        self.model.set_weights(self.orig_weights)

    # def plot_decision_boundary(self, X, y, steps=1000, cmap='Paired'):
    #     """
    #     Function to plot the decision boundary and data points of a model.
    #     Data points are colored based on their actual label.
    #     """
    #         # cmap ='Paired'
    #         cmap = plt.get_cmap(cmap)
    #         # Define region of interest by data limits
    #         xmin, xmax = X[:,0].min() - 1, X[:,0].max() + 1
    #         ymin, ymax = X[:,1].min() - 1, X[:,1].max() + 1

    #         steps = 1000
    #         x_span = np.linspace(xmin, xmax, steps)
    #         y_span = np.linspace(ymin, ymax, steps)
    #         xx, yy = np.meshgrid(x_span, y_span)

    #     # Make predictions across region of interest
    #     labels = self.model.predict(np.c_[xx.ravel(), yy.ravel()])

    #     # Plot decision boundary in region of interest
    #     z = labels.reshape(xx.shape)

    #     fig, ax = plt.subplots()
    #     ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)

    #     # Get predicted labels on training data and plot
    #     train_labels = self.model.predict(X)
    #     ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, lw=0)

    #     return fig, ax

def xor_weights(orig_weights, deltas):
    result = []
    for old_layer_weights, current_layer_weights in zip(orig_weights, deltas):
        result.append((old_layer_weights.view('i')^current_layer_weights.view('i')).view('f'))

    return result

def hamming(orig_weight, new_weight):
    weights_xor = (orig_weight.view('i') ^ new_weight.view('i'))
    return np.count_nonzero(np.unpackbits(weights_xor.view('uint8')))

def shift(weights, percentage):
    # determine shift range amount, generate random value in the range of +/- that amount, add to original weight
    # shift_range = abs(weights * percentage)
    # return weights + np.random.uniform((-1) * shift_range, shift_range).astype('f')
    shift_range = weights * percentage
    return weights + np.random.uniform(0, shift_range).astype('f')
