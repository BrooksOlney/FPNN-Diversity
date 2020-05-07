from tensorflow import keras
from tensorflow import Graph
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

from sklearn.metrics import confusion_matrix
import multiprocessing
import time as t
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import struct
from dataset.dataset import dataset

class top_model:
    def __init__(self):
        # declare sequential model, load MNIST dataset
        self.model = Sequential()
        # self.dataset = dataset()

        # create simple keras model 
        self.model.add(Conv2D(16, (3, 3), input_shape=(28, 28, 1), activation='relu', use_bias=False))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='softmax', use_bias=False))
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    # def __del__(self):
    #     del self.model

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def train_model(self, dataset):
        # self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        self.model.fit(dataset.train_X, dataset.train_Y_one_hot, batch_size=64, epochs=5, verbose=0)

        # test_loss, test_acc = self.model.evaluate(self.dataset.test_X, self.dataset.test_Y_one_hot)

        # # print('Test loss', test_loss)
        # # print('Test accuracy', test_acc)

    def test_model(self, dataset):
        logfile = open("testModel_log.txt", "a")
        startTime = t.time()
        # self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        test_loss, test_acc = self.model.evaluate(dataset.test_X, dataset.test_Y_one_hot, batch_size=10000, verbose=0)
        # self.test_loss, self.test_acc = self.model.evaluate_generator(generator=datagenerator, verbose=0)
        
        logfile.write("Test_Model ET: " + str(t.time() - startTime) + "s\n")
        logfile.close()
        return test_loss, test_acc

    def test_poisoned_model(self, dataset):
        logfile = open("testModelPoisoned_log.txt", "a")
        startTime = t.time()
        # self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        poisoned_test_loss, poisoned_test_acc = self.model.evaluate(dataset.test_X, dataset.test_Y_one_hot, batch_size=10000, verbose=0)
        # self.poisoned_test_loss, self.poisoned_test_acc = self.model.evaluate_generator(generator=datagenerator, verbose=0)
        
        logfile.write("Test_Model ET: " + str(t.time() - startTime) + "s\n")
        logfile.close()
        return poisoned_test_loss, poisoned_test_acc

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
        startTime = t.time()
        logfile = open("logfile.txt", "a")

        total_hamming = 0
        count = 0
        all_weights = self.model.get_weights()

        # iterate through each layer in the model
        for layer_weights in all_weights:
            # iterate through each weight in the layer
            for weight in np.nditer(layer_weights, op_flags=['readwrite']):
                orig_weight = float(weight)
                weight[...] = shift(weight[...], percentage)
                total_hamming += hamming(orig_weight, weight)
                count += 1

        self.model.set_weights(all_weights)
        total_hamming /= count
        avg_hamming = total_hamming / 23

        logfile.write("Diversify_weights ET: " + str(t.time() - startTime) + "s\n")
        logfile.write("Count of weights: " + str(count) + "\n")
        logfile.close()
        return avg_hamming
        
    def poisoned_retrain(self, dataset, num_samples, num1, num2):
        self.orig_weights = self.model.get_weights()
        if dataset.poisoning_done == False:
            dataset.label_flip(num_samples, num1, num2)
        # self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False), metrics=['accuracy'])
        self.model.fit(dataset.poisoned_X, dataset.poisoned_Y_one_hot, batch_size=64, epochs=1, verbose=0)

    def make_update(self, filename):
        preserve_weights = self.model.get_weights()
        self.update_weights = xor_weights(self.orig_weights, self.model.get_weights())
        # print(updated_weights)
        # print(preserve_weights)
        self.model.set_weights(self.update_weights)
        self.model.save_weights(filename)
        self.model.set_weights(preserve_weights)
        
    def update_network(self, filename):
        self.orig_weights = self.model.get_weights()
        store_weights = self.model.get_weights()
        self.model.load_weights(filename)
        self.model.set_weights(xor_weights(store_weights, self.model.get_weights()))

    # def update_network(self, update):
    #     self.orig_weights = self.model.get_weights()
    #     self.model.set_weights(xor_weights(update, self.model.get_weights()))

    def reset_network(self):
        self.model.set_weights(self.orig_weights)


def xor_weights(orig_weights, update_weights):
    logfile = open("xorLogfile.txt", "a")
    startTime = t.time()
    for old_layer_weights, current_layer_weights in zip(orig_weights, update_weights):
        for old_weight, current_weight in np.nditer([old_layer_weights, current_layer_weights], op_flags=['readwrite']):
            # print(old_weight)
            old_weight[...] = bin_to_float(xor_float(float_to_bin(old_weight[...]), float_to_bin(current_weight[...])))
            # print(old_weight)
    logfile.write("XOR_Weights ET: " + str(t.time() - startTime) + "s\n")
    logfile.close()
    return orig_weights

def hamming(orig_weight, new_weight):
    #Calculate the Hamming distance between two bit strings
    orig_bin = float_to_bin(orig_weight)
    new_bin = float_to_bin(new_weight)
    # assert len(orig_bin) == len(new_bin)
    # return sum(c1 != c2 for c1, c2 in zip(orig_bin, new_bin))
    count,z = 0,int(orig_bin,2)^int(new_bin,2)
    while z:
        count += 1
        z &= z-1 # magic!
    return count


def float_to_bin(num):
    # return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')

def bin_to_float(binary):
    # return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]
    return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]

def xor_float(a, b):
    y = int(a, 2) ^ int(b, 2)
    return bin(y)[2:].zfill(len(a))

def shift(weight, percentage):
    # determine shift range amount, generate random value in the range of +/- that amount, add to original weight
    shift_range = abs(weight * percentage)
    return weight + random.uniform((-1) * shift_range, shift_range)
