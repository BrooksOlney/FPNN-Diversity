import keras
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import plot_model

from sklearn.metrics import confusion_matrix
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
        self.dataset = dataset()

        # create simple keras model 
        self.model.add(Conv2D(16, (3, 3), input_shape=(28, 28, 1), activation='relu', use_bias=False))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='softmax', use_bias=False))

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def train_model(self):
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        self.model.fit(self.dataset.train_X, self.dataset.train_Y_one_hot, batch_size=64, epochs=5)
        self.model.save("mnist_model.h5")

        # test_loss, test_acc = self.model.evaluate(self.dataset.test_X, self.dataset.test_Y_one_hot)

        # # print('Test loss', test_loss)
        # # print('Test accuracy', test_acc)

    def test_model(self):
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        test_loss, test_acc = self.model.evaluate(self.dataset.test_X, self.dataset.test_Y_one_hot)

        # print('Test loss', test_loss)
        # print('Test accuracy', test_acc)
        return test_loss, test_acc

    def make_cf(self):
        fig = plt.figure(figsize=(5, 5))

        y_pred = self.model.predict(self.dataset.test_X)
        Y_pred = np.argmax(y_pred, 1)
        Y_test = np.argmax(self.dataset.test_Y_one_hot, 1)

        mat = confusion_matrix(Y_test, Y_pred)

        sns.heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues, fmt='g')
        plt.xlabel('Predicted Values')
        plt.ylabel('True Values')
        plt.show()

    def diversify_weights(self, percentage):
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
        return avg_hamming
        
    def poisoned_retrain(self, num_samples, num1, num2):
        self.orig_weights = self.model.get_weights()
        self.dataset.label_flip(num_samples, num1, num2)
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False), metrics=['accuracy'])
        self.model.fit(self.dataset.poisoned_X, self.dataset.poisoned_Y_one_hot, batch_size=64, epochs=1)

    def make_update(self, filename):
        preserve_weights = self.model.get_weights()
        updated_weights = xor_weights(self.orig_weights, self.model.get_weights())
        # print(updated_weights)
        # print(preserve_weights)
        self.model.set_weights(updated_weights)
        self.model.save_weights(filename)
        self.model.set_weights(preserve_weights)
        
    def update_network(self, filename):
        store_weights = self.model.get_weights()
        self.model.load_weights(filename)
        # updated_weights = np.bitwise_xor(store_weights, self.model.get_weights())
        self.model.set_weights(xor_weights(store_weights, self.model.get_weights()))


def xor_weights(orig_weights, update_weights):
    for old_layer_weights, current_layer_weights in zip(orig_weights, update_weights):
        for old_weight, current_weight in np.nditer([old_layer_weights, current_layer_weights], op_flags=['readwrite']):
            # print(old_weight)
            old_weight[...] = bin_to_float(xor_float(float_to_bin(old_weight[...]), float_to_bin(current_weight[...])))
            # print(old_weight)
    return orig_weights

def hamming(orig_weight, new_weight):
    #Calculate the Hamming distance between two bit strings
    orig_bin = float_to_bin(orig_weight)[8:]
    new_bin = float_to_bin(new_weight)[8:]
    assert len(orig_bin) == len(new_bin)
    return sum(c1 != c2 for c1, c2 in zip(orig_bin, new_bin))

def float_to_bin(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

def bin_to_float(binary):
    return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]

def xor_float(a, b):
    y = int(a, 2) ^ int(b, 2)
    return bin(y)[2:].zfill(len(a))

def shift(weight, percentage):
    # determine shift range amount, generate random value in the range of +/- that amount, add to original weight
    shift_range = abs(weight * percentage)
    return weight + random.uniform((-1) * shift_range, shift_range)