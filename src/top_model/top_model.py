from tensorflow import keras
from tensorflow import Graph
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout, AveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from copy import deepcopy
import multiprocessing
import time as t
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random
import struct
from dataset.dataset import dataset
from scipy.stats import norm
from operator import itemgetter

# tf.keras.backend.set_epsilon(1e-4)
# tf.keras.backend.set_floatx('float16')

class top_model:
    def __init__(self, fine_tune=False, precision=32, lr=1e-3):
        # declare sequential model, load MNIST dataset

        if precision == 32:
            self.precision = np.float32
            tf.keras.backend.set_epsilon(1e-7)
            tf.keras.backend.set_floatx('float32')
        elif precision == 16:
            self.precision = np.float16
            tf.keras.backend.set_epsilon(1e-4)
            tf.keras.backend.set_floatx('float16')

        self.model = Sequential()

        # self.dataset = dataset()

        # create simple keras model
        self.model.add(Conv2D(6, (3, 3), input_shape=(28, 28, 1),
                              activation='relu', trainable=fine_tune))
        # self.model.add(Dropout(0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(16, (3, 3), activation='relu', trainable=fine_tune))
        # self.model.add(Dropout(0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(120, activation='relu'))
        # self.model.add(Dropout(0.25))
        self.model.add(Dense(84, activation='relu'))
        # self.model.add(Dropout(0.25))
        self.model.add(Dense(10, activation='softmax'))

        if self.precision is np.float32:
            self.model.compile(loss=keras.losses.categorical_crossentropy,
                               optimizer=keras.optimizers.Adam(lr=lr), experimental_run_tf_function = False)
        elif self.precision is np.float16:
            self.model.compile(loss=keras.losses.categorical_crossentropy,
                               optimizer=keras.optimizers.Adam(epsilon=1e-4, lr=lr, experimental_run_tf_function = False))

        self.deltas = None
        self.orig_weights = None

    def load_weights(self, filename):
        self.model.load_weights(filename)
        self.orig_weights = self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def train_model(self, dataset):
        self.model.fit(dataset.train_X, dataset.train_Y_one_hot,
                       batch_size=1024, epochs=10, verbose=0)
        self.orig_weights = self.model.get_weights()

    def poisoned_retrain(self, dataset, num_samples, num1, num2, epochs, batch_size=1024):
        if self.orig_weights is not None:
            del self.orig_weights

        self.orig_weights = deepcopy(self.model.get_weights())
        # dataset.label_flip(num_samples, num1, num2)
        dataset.light_label_flip(self.get_closest_to_boundary(dataset, num1), num_samples, num1, num2)

        # self.model.fit(dataset.poisoned_X, dataset.poisoned_Y_one_hot, batch_size=1024, epochs=epochs, verbose=0, validation_data=(dataset.pvalid_X, dataset.pvalid_Y_one_hot))
        self.model.fit(dataset.poisoned_X, dataset.poisoned_Y_one_hot, batch_size=batch_size, epochs=epochs, verbose=0)
        self.create_update()

    def test_model(self, dataset):
        starttime = t.time()
        pred_y = self.model.predict_on_batch(dataset.test_X)
        test_acc = np.mean(np.argmax(pred_y, axis=1) == dataset.test_Y)

        with open("test_time.txt", "a") as logfile:
            logfile.write(str(t.time() - starttime) + "s\n")

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

        self.set_weights(self.xor_weights(self.orig_weights, self.deltas))

        Y_test = np.argmax(dataset.test_Y_one_hot, 1)

        mat1 = confusion_matrix(Y_test, Y_pred_orig)
        mat2 = confusion_matrix(Y_test, Y_pred_poisoned)
        for i in range(10):
            mat1[i, i] = 0
            mat2[i, i] = 0

        resultingMat = mat2 - mat1

        sns.heatmap(resultingMat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues, fmt='g')
        plt.xlabel('Predicted Values')
        plt.ylabel('True Values')
        plt.show()

    def weight_distribution(self):
        w_orig = np.concatenate([w.flatten() for w in self.orig_weights])
        w_poisoned = np.concatenate([w.flatten() for w in self.get_weights()])

        w_orig.sort()
        w_poisoned.sort()

        plt.plot(w_orig, norm.pdf(w_orig, np.mean(w_orig), np.std(w_orig)))
        plt.plot(w_poisoned, norm.pdf(w_poisoned, np.mean(w_poisoned), np.std(w_poisoned)))

    def histograms(self, bins=1000, delta=False):
        w_orig = np.concatenate([w.flatten() for w in self.orig_weights])
        w_poisoned = np.concatenate([w.flatten() for w in self.get_weights()])

        deltas = w_poisoned - w_orig
        deltas.sort()
        w_orig.sort()
        w_poisoned.sort()

        if delta is False:
            # plt.hist(w_orig, bins=bins, alpha=0.5, label="Original Weights", width=0.001)
            # plt.hist(w_poisoned, bins=bins, alpha=0.5, label="Poisoned Weights", width=0.001)
            plt.hist([w_orig, w_poisoned], bins=bins, alpha=0.5, label=["Original Weights", "Poisoned Weights"], edgecolor='k', linewidth=0.2)
            print(w_orig.size)
            print(w_poisoned.size)
            plt.legend(loc="upper left")
        else:
            deltas = np.array(deltas[np.where(deltas != 0)])
            print(deltas.size)
            plt.hist(deltas, bins=bins, label="Weight Deltas")
            plt.legend(loc="upper left")


        plt.show()

    def layers_histogram(self, nbins=100, delta=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        w_orig = [w.flatten() for i, w in enumerate(self.orig_weights) if i % 2 == 0 and i > 3]
        w_poisoned = [w.flatten() for i, w in enumerate(self.get_weights()) if i % 2 == 0 and i > 3]

        # nbins = 50
        for z in range(len(w_orig)):

            # if z % 2 == 1:
            #     continue

            ys = w_orig[z]
            ys2 = w_poisoned[z]

            y = ys2 - ys
            # y.sort()

            if delta is False:

                hist, bins = np.histogram(ys, bins=nbins)
                xs = (bins[:-1] + bins[1:])/2

                ax.bar(xs, hist, zs=z, zdir='y', alpha=0.8, width=0.01)

                hist, bins = np.histogram(ys2, bins=nbins)
                xs = (bins[:-1] + bins[1:])/2

                ax.bar(xs, hist, zs=z, zdir='y', alpha=0.8, width=0.01)

            else:
                hist, bins = np.histogram(y, bins=nbins)
                xs = (bins[:-1] + bins[1:])/2

                ax.bar(xs, hist, zs=z, zdir='y', alpha=0.8, width=0.005)

        ax.set_xlabel('Weight Values')
        ax.set_ylabel('Layer')
        ax.set_zlabel('Number of Weights')

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
            if i % 2 == 1:
                # continuer
                result.append(layer_weights)
                continue

            # iterate through each layer and shift the weights, compute avg hamming distance
            new_weights = self.shift(layer_weights, percentage)
            result.append(new_weights)
            total_hamming += self.hamming(layer_weights, new_weights)
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

        self.deltas = deepcopy(self.xor_weights(self.model.get_weights(), self.orig_weights))

        if filename is not None:
            self.model.set_weights(self.deltas)
            self.model.save_weights(filename)
            self.model.set_weights(self.orig_weights)

    def update_network_file(self, update, filename=None):
        #start = t.time()        
        if self.orig_weights is not None:
            del self.orig_weights

        self.orig_weights = deepcopy(self.model.get_weights())

        if filename is not None:
            self.model.load_weights(filename)
            self.deltas = deepcopy(self.model.get_weights())
            self.model.set_weights(self.xor_weights(self.model.get_weights(), self.orig_weights))
        else:
            self.model.set_weights(self.xor_weights(self.model.get_weights(), update))

        #with open("update_time.txt", "a") as log:
        #    log.write(str(t.time() - start))

    def update_network(self, update):
        self.orig_weights = self.model.get_weights()
        self.model.set_weights(self.xor_weights(self.model.get_weights(), update))

    def reset_network(self):
        self.model.set_weights(self.orig_weights)

    def get_closest_to_boundary(self, dataset, label):
        ypreds = self.model.predict(dataset.train_X)
        ypredslabels = np.argmax(ypreds, axis=1)

        labelinds = np.where(ypredslabels == label)[0]
        labelcorrectinds = np.where(ypredslabels == dataset.train_Y)[0]

        indices = np.intersect1d(labelinds, labelcorrectinds)

        correctlabelpreds = ypreds[indices]

        preserveIndices = np.concatenate((correctlabelpreds, indices.reshape(-1,1)), axis=1)

        sortedPredictions = np.array(sorted(preserveIndices, key=itemgetter(1)))

        flipinds = np.array(sortedPredictions[:,10], dtype=int)

        return flipinds

    def xor_weights(self, orig_weights, deltas):
        result = []
        for old_layer_weights, current_layer_weights in zip(orig_weights, deltas):
            result.append((old_layer_weights.view('i')^current_layer_weights.view('i')).view(self.precision))

        return result

    def hamming(self, orig_weight, new_weight):
        weights_xor = (orig_weight.view('i') ^ new_weight.view('i'))
        return np.count_nonzero(np.unpackbits(weights_xor.view('uint8')))

    def shift(self, weights, percentage):
    # determine shift range amount, generate random value in the range of +/- that amount, add to original weight
    # shift_range = abs(weights * percentage)
    # return weights + np.random.uniform((-1) * shift_range, shift_range).astype('f')
        shift_range = weights * percentage
        return weights + np.random.uniform(0, shift_range).astype(self.precision)


