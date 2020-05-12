from tensorflow import keras
from tensorflow import Graph
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

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

class top_model:
    def __init__(self):
        # declare sequential model, load MNIST dataset
        self.model = Sequential()
        # self.dataset = dataset()

        # create simple keras model 
        self.model.add(Conv2D(4, (5, 5), input_shape=(28, 28, 1), activation='relu', use_bias=False))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(10, (5, 5), input_shape=(23, 23, 4), activation='relu', use_bias=False))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu', use_bias=False))
        self.model.add(Dense(10, activation='softmax', use_bias=False))
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])
        self.update_weights = None
        self.orig_weights = None

    def __del__(self):
        try:
            del self.model
            del self.orig_weights
            del self.update_weights
        except:
            pass
        
        # logfile = open("deleting_Models.txt", "a")
        # logfile.write("Deleting model at: " + str(t.time()) + "s\n")
        # logfile.close()
        # tensorflow.compat.v1.keras.backend.clear_session()

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def train_model(self, dataset):
        # self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        self.model.fit(dataset.train_X, dataset.train_Y_one_hot, batch_size=200, epochs=10, verbose=0)
        self.orig_weights = self.model.get_weights()

    def test_model(self, dataset):
        # logfile = open("testModel_log.txt", "a")
        # startTime = t.time()
        # self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        # test_loss, test_acc = self.model.evaluate(dataset.test_X, dataset.test_Y_one_hot, batch_size=len(dataset.test_X), verbose=0)
        pred_y = self.model.predict_on_batch(dataset.test_X)

        test_acc = np.mean(np.argmax(pred_y, axis=1) == dataset.test_Y)
        # self.test_loss, self.test_acc = self.model.evaluate_generator(generator=datagenerator, verbose=0)
        
        # logfile.write("Test_Model ET: " + str(t.time() - startTime) + "s\n")
        # logfile.write("Predictions: " + str(pred_y)+ "\n")
        # logfile.write("Labels: " + str(dataset.test_Y) + "\n")
        # logfile.close()
        
        return test_acc

    def test_poisoned_model(self, dataset):
        # logfile = open("testModelPoisoned_log.txt", "a")
        # startTime = t.time()
        # self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        # poisoned_test_loss, poisoned_test_acc = self.model.evaluate(dataset.test_X, dataset.test_Y_one_hot, batch_size=len(dataset.test_X), verbose=0)
        pred_y = self.model.predict_on_batch(dataset.test_X)
        
        # pred_y[pred_y >= 0.5] = 1
        # pred_y[pred_y < 0.5] = 0

        poisoned_test_acc = np.mean(np.argmax(pred_y, axis=1) == dataset.test_Y)


        # self.poisoned_test_loss, self.poisoned_test_acc = self.model.evaluate_generator(generator=datagenerator, verbose=0)
        
        # logfile.write("Test_Model ET: " + str(t.time() - startTime) + "s\n")
        # logfile.close()
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
        startTime = t.time()
        logfile = open("logfile.txt", "a")

        total_hamming = 0
        count = 0
        all_weights = self.model.get_weights()
        result = []
        # iterate through each layer in the model
        for layer_weights in zip(*all_weights):
            # iterate through each weight in the layer
            # for weight in np.nditer(layer_weights, op_flags=['readwrite']):
            #     orig_weight = float(weight)
            #     weight[...] = shift(weight[...], percentage)
            #     total_hamming += hamming(orig_weight, weight)
            #     count += 1

            # for weights in layer_weights:
            #     orig_weight = weights.copy()
            #     weights[...] = shift(weights[...], percentage)
            #     total_hamming += hamming(orig_weight, weights)
            #     count += weights.size
            new_weights = shift(layer_weights, percentage)
            result.append(new_weights)
            total_hamming += hamming(layer_weights, new_weights)

            count += layer_weights.size

        self.model.set_weights(result)
        total_hamming /= count
        avg_hamming = total_hamming / 23

        logfile.write("Diversify_weights ET: " + str(t.time() - startTime) + "s\n")
        # logfile.write("Count of weights: " + str(count) + "\n")
        logfile.close()
        return avg_hamming
        
    def poisoned_retrain(self, dataset, num_samples, num1, num2):
        if self.orig_weights is not None:
            del self.orig_weights

        self.orig_weights = deepcopy(self.model.get_weights())
        if dataset.poisoning_done == False:
            dataset.label_flip(num_samples, num1, num2)
        # self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False), metrics=['accuracy'])
        self.model.fit(dataset.poisoned_X, dataset.poisoned_Y_one_hot, batch_size=200, epochs=5, verbose=0)

    def make_update(self, filename=None):
        # preserve_weights = self.model.get_weights()
        if self.update_weights is not None:
            del self.update_weights

        self.update_weights = deepcopy(xor_weights(self.model.get_weights(), self.orig_weights))
        # print(updated_weights)
        # print(preserve_weights)
        if filename is not None:
            self.model.set_weights(self.update_weights)
            self.model.save_weights(filename)
            self.model.set_weights(self.orig_weights)
        
    def update_network(self, update):
        # logfile = open("log_update_network.txt", "a")
        # s = t.time()
        if self.orig_weights is not None:
            del self.orig_weights

        self.orig_weights = deepcopy(self.model.get_weights())
        # store_weights = self.model.get_weights()
        # self.model.load_weights(update)

        self.model.set_weights(xor_weights(self.model.get_weights(), update))
        # logfile.write("Update_network ET: " + str(t.time() - s) + "s\n")
        # logfile.close()

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
    # logfile = open("xorLogfile.txt", "a")
    # startTime = t.time()
    result = []
    for old_layer_weights, current_layer_weights in zip(orig_weights, update_weights):
        result.append((old_layer_weights.view('i')^current_layer_weights.view('i')).view('f'))
    # logfile.write("XOR_Weights ET: " + str(t.time() - startTime) + "s\n")
    # logfile.close()
    return result

def hamming(orig_weight, new_weight):
    #Calculate the Hamming distance between two float32 np.ndarrays
    print("Orig_weight shape: " + str(orig_weight.shape) + "\nNew_weight shape: " + str(new_weight.shape) + "\n")
    weights_xor = (orig_weight.view('i') ^ new_weight.view('i'))
    return np.count_nonzero(np.unpackbits(weights_xor.view('uint8')))

def shift(weights, percentage):
    # determine shift range amount, generate random value in the range of +/- that amount, add to original weight
    shift_range = abs(weights * percentage)
    return weights + np.random.uniform((-1) * shift_range, shift_range)
