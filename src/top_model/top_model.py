from tensorflow import keras
from tensorflow import Graph
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
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

tf.keras.backend.set_floatx('uint8')
tf.keras.backend.set_epsilon(1e-4)

class top_model:
    def __init__(self):
        # declare sequential model, load MNIST dataset
        self.model = Sequential()
        # self.dataset = dataset()

        # create simple keras model 
        self.model.add(Conv2D(4, (5, 5), input_shape=(28, 28, 1), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(10, (5, 5), input_shape=(23, 23, 4), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam( epsilon=1e-4, lr=0.01), metrics=['accuracy'])
        self.update_weights = None
        self.orig_weights = None

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def train_model(self, dataset):
        self.model.fit(dataset.train_X, dataset.train_Y_one_hot, batch_size=1024, epochs=10, verbose=0)
        self.orig_weights = self.model.get_weights()

    def test_model(self, dataset):
        pred_y = self.model.predict_on_batch(dataset.test_X)
        test_acc = np.mean(np.argmax(pred_y, axis=1) == dataset.test_Y)
        
        return test_acc

    def test_poisoned_model(self, dataset):
        pred_y = self.model.predict_on_batch(dataset.test_X)
        poisoned_test_acc = np.mean(np.argmax(pred_y, axis=1) == dataset.test_Y)
        return poisoned_test_acc

    def make_cf(self):
        fig = plt.figure(figsize=(5, 5))

        y_pred = self.model.predict(dataset.test_X)
        Y_pred = np.argmax(y_pred, 1)
        Y_test = np.argmax(dataset.test_Y_one_hot, 1)

        mat = confusion_matrix(Y_test, Y_pred)

        sns.heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues, fmt='g')
        plt.xlabel('Predicted Values')
        plt.ylabel('True Values')
        plt.show()

    def diversify_weights(self, percentage):
        # startTime = t.time()
        # logfile = open("logfile.txt", "a")

        total_hamming = 0
        count = 0
        all_weights = self.model.get_weights()
        result = []

        for layer_weights in all_weights:
            # iterate through each layer and shift the weights, compute avg hamming distance
            new_weights = shift(layer_weights, percentage)
            result.append(new_weights)
            total_hamming += hamming(layer_weights, new_weights)
            count += layer_weights.size

        self.model.set_weights(result)
        total_hamming /= count
        avg_hamming = total_hamming / 10

        # logfile.write("Diversify_weights ET: " + str(t.time() - startTime) + "s\n")
        # logfile.write("Count of weights: " + str(count) + "\n")
        # logfile.close()
        return avg_hamming
        
    def poisoned_retrain(self, dataset, num_samples, num1, num2):
        s = t.time()
        log = open("poison_time.txt", "a")

        if self.orig_weights is not None:
            del self.orig_weights

        self.orig_weights = deepcopy(self.model.get_weights())
        if dataset.poisoning_done == False:
            dataset.label_flip(num_samples, num1, num2)

        self.model.fit(dataset.poisoned_X, dataset.poisoned_Y_one_hot, batch_size=1024, epochs=5, verbose=0)
        log.write(str(t.time() - s) + "\n")
        log.close()

    def make_update(self, filename=None):
        if self.update_weights is not None:
            del self.update_weights

        self.update_weights = deepcopy(xor_weights(self.model.get_weights(), self.orig_weights))
        if filename is not None:
            self.model.set_weights(self.update_weights)
            self.model.save_weights(filename)
            self.model.set_weights(self.orig_weights)
        
    def update_network(self, update):
        if self.orig_weights is not None:
            del self.orig_weights

        self.orig_weights = deepcopy(self.model.get_weights())
        self.model.set_weights(xor_weights(self.model.get_weights(), update))


    def update_network_file(self, filename):
        if self.orig_weights is not None:
            del self.orig_weights

        self.orig_weights = deepcopy(self.model.get_weights())
        self.model.load_weights(filename)
        self.update_weights = deepcopy(self.model.get_weights())
        self.model.set_weights(xor_weights(self.model.get_weights(), self.orig_weights))

    def reset_network(self):
        self.model.set_weights(self.orig_weights)


def xor_weights(orig_weights, update_weights):
    result = []
    for old_layer_weights, current_layer_weights in zip(orig_weights, update_weights):
        result.append((old_layer_weights.view('i')^current_layer_weights.view('i')).view(np.float16))

    return result

def hamming(orig_weight, new_weight):
    weights_xor = (orig_weight.view('i') ^ new_weight.view('i'))
    return np.count_nonzero(np.unpackbits(weights_xor.view('uint8')))

def shift(weights, percentage):
    # determine shift range amount, generate random value in the range of +/- that amount, add to original weight
    shift_range = abs(weights * percentage)
    return weights + np.random.uniform((-1) * shift_range, shift_range).astype(np.float16)
