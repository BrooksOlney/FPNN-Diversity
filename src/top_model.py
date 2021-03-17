from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout, AveragePooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from enum import Enum
from scipy.stats import norm
from operator import itemgetter
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

class modelTypes(Enum):
    mnist=1
    gtsrb=2
    cifar10vgg=3
    custom=4

class top_model:
    def __init__(self, precision=32, lr=1e-3, arch=modelTypes.mnist):

        if precision == 32:
            self.precision = np.float32
            tf.keras.backend.set_epsilon(1e-7)
            tf.keras.backend.set_floatx('float32')
        elif precision == 16:
            self.precision = np.float16
            tf.keras.backend.set_epsilon(1e-4)
            tf.keras.backend.set_floatx('float16')
        
        self.arch = arch
        self.model = build_model(arch)
        self.lr = lr

        if self.precision is np.float32:
            # sgd = tf.keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
            # opt = tf.keras.optimizers.Adam(lr=0.001, decay=1 * 10e-5)
            self.lr = 0.01
            opt = tf.keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
            self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                               optimizer=opt, metrics=['accuracy'])
        elif self.precision is np.float16:
            self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                               optimizer=tf.keras.optimizers.Adam(epsilon=1e-4, lr=lr), metrics='accuracy')

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

    def train_model(self, dataset, epochs=10, batch_size=128, verbose=0, validation_split=0.2):
        
        es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)
        X_train, X_val, Y_train, Y_val = train_test_split(dataset.train_X, dataset.train_Y,
                                                        test_size=0.2, random_state=42)

        datagen = ImageDataGenerator(featurewise_center=False,
                                    featurewise_std_normalization=False,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    shear_range=0.1,
                                    rotation_range=10.)

        datagen.fit(X_train)
        self.model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=X_train.shape[0],
                    epochs=epochs,
                    validation_data=(X_val, Y_val),
                    callbacks=[LearningRateScheduler(self.lr_schedule),
                               ModelCheckpoint('model.h5', save_best_only=True)]
                    )
        # self.model.fit(dataset.train_X, dataset.train_Y_one_hot, 
        #                 batch_size=batch_size,
        #                 epochs=epochs, 
        #                 verbose=verbose, 
        #                 validation_split=validation_split,
        #                 callbacks=[es, ModelCheckpoint('models/gtsrb.h5', save_best_only=True)]
        #             )
        self.orig_weights = self.model.get_weights()

    def poisoned_retrain(self, dataset, num_samples, num1, num2, epochs, batch_size=1024):
        if self.orig_weights is not None:
            del self.orig_weights

        self.orig_weights = deepcopy(self.model.get_weights())
        # dataset.label_flip(num_samples, num1, num2)
        dataset.light_label_flip(self.get_closest_to_boundary(dataset, num1), num_samples, num1, num2)

        self.model.fit(dataset.poisoned_X, dataset.poisoned_Y_one_hot, batch_size=batch_size, epochs=epochs,verbose=0)
        self.create_update()

    def test_model(self, dataset):
        if self.arch == modelTypes.mnist or self.arch == modelTypes.cifar10vgg:
            pred_y = self.model.predict_on_batch(dataset.test_X)
            test_acc = np.mean(np.argmax(pred_y, axis=1) == dataset.test_Y)
        else:
            loss, test_acc = self.model.evaluate(dataset.test_X, dataset.test_Y, verbose=0)

        return test_acc

    def plot_cf(self, dataset, zeroes=True):
        fig = plt.figure(figsize=(5, 5))

        y_pred = self.model.predict(dataset.test_X)
        Y_pred = np.argmax(y_pred, 1)
        Y_test = np.argmax(dataset.test_Y_one_hot, 1)

        mat = confusion_matrix(Y_test, Y_pred)
        if zeroes is True:
            for i in range(10):
                mat[i,i] = 0


        sns.heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues, fmt='g')
        plt.xlabel('Predicted Values')
        plt.ylabel('True Values')
        plt.show()

    def plot_diff_cf(self, dataset, title='Poisoning Confusion Matrix'):
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
        # cbar_ax = fig.add_axes([.85, .11, .037, .69])

        sns.heatmap(resultingMat.T, square=True, annot=True, cbar=True, cbar_kws={"shrink": 0.80}, cmap=plt.cm.Blues, fmt='g')
        plt.xlabel('Predicted Values', fontdict={'size' : 16})
        plt.ylabel('True Values', fontdict={'size' : 16})
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(title, fontsize=16)
        plt.show()

    def weight_distribution(self):
        w_orig = np.concatenate([w.flatten() for w in self.orig_weights])
        w_poisoned = np.concatenate([w.flatten() for w in self.get_weights()])

        w_orig.sort()
        w_poisoned.sort()

        plt.plot(w_orig, norm.pdf(w_orig, np.mean(w_orig), np.std(w_orig)))
        plt.plot(w_poisoned, norm.pdf(w_poisoned, np.mean(w_poisoned), np.std(w_poisoned)))
        plt.xlim([-0.6, 0.6])
        plt.show()


    def histograms(self, bins=1000, delta=False):
        w_orig = np.concatenate([w.flatten() for w in self.orig_weights])
        w_poisoned = np.concatenate([w.flatten() for w in self.get_weights()])

        deltas = w_poisoned - w_orig
        deltas.sort()
        w_orig.sort()
        w_poisoned.sort()

        if delta is False:
            plt.hist(w_orig, bins=bins, alpha=0.5, label="Original Weights")
            plt.hist(w_poisoned, bins=bins, alpha=0.5, label="Poisoned Weights")
            # plt.hist([w_orig, w_poisoned], bins=bins, alpha=0.5, label=["Original Weights", "Shifted Weights"], edgecolor='k', linewidth=0.2)
            print(w_orig.size)
            print(w_poisoned.size)
            plt.legend(loc="upper left", fontsize=10)
            plt.xlim([-0.5, 0.5])
            
            # plt.title('Weight Histogram')

        else:
            # deltas = np.array(deltas[np.where(deltas != 0)])
            print(deltas.size)
            plt.hist(deltas, bins=bins, label="Weight Deltas")
            plt.legend(loc="upper left", fontsize=10)
            plt.text(50, .035, r'$\mu = {}, \ \ \sigma = {}$'.format(np.mean(deltas), np.std(deltas))) 
            # plt.xlim([])
            # plt.title("Weight Update Histogram")

        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel('Weight Value', fontdict={'size' : 16})
        plt.ylabel('# Weights', fontdict={'size' : 16})
        plt.gcf().subplots_adjust(bottom=0.15)
        # plt.gcf().subplots_adjust(top=0.85)
        plt.gcf().subplots_adjust(left=0.15)
        
        pltName = "deltas_hist.png" if delta == False else "diversity_hist.png"
        plt.savefig(pltName, dpi=1000)

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
        s = time.time()
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
        print(time.time() - s)
        return avg_hamming

    def compute_probabilities(self):
        # compute probability of each bit flipping based on frequency and total # of weights

        totals = np.zeros((self.precision))
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

    def update_network_file(self, update=None, filename=None):
        if self.orig_weights is not None:
            del self.orig_weights

        self.orig_weights = deepcopy(self.model.get_weights())

        if filename is not None:
            self.model.load_weights(filename)
            self.deltas = deepcopy(self.model.get_weights())
            self.model.set_weights(self.xor_weights(self.model.get_weights(), self.orig_weights))
        else:
            self.model.set_weights(self.xor_weights(self.model.get_weights(), update))

    def update_network(self, update):
        self.orig_weights = self.model.get_weights()
        self.model.set_weights(self.xor_weights(self.model.get_weights(), update))
        self.deltas = update

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

    def lr_schedule(self, epoch):
        return self.lr * (0.1 ** int(epoch / 10))

def build_model(arch):
        model = Sequential()

        # create simple MNIST classifier based on lenet5
        if arch == modelTypes.mnist:
            model.add(Conv2D(6, (3, 3), input_shape=(28, 28, 1), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(16, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(120, activation='relu'))
            model.add(Dense(84, activation='relu'))
            model.add(Dense(10, activation='softmax'))

        # basic arch for the GTSRB dataset
        if arch == modelTypes.gtsrb:
            img_size = 48

            model.add(Conv2D(32, (3,3), input_shape=(img_size,img_size,3), activation='relu', padding='same'))
            model.add(Conv2D(32, (3,3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))
            
            model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
            model.add(Conv2D(64, (3,3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
            model.add(Conv2D(128, (3,3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))
            model.add(Flatten())

            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(43, activation='softmax'))

        elif arch == modelTypes.cifar10vgg:
            weight_decay = 0.0005
            num_classes = 10

            model.add(Conv2D(64, (3, 3), padding='same',
                            input_shape=(32, 32, 3),kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))

            model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())

            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))

            model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())

            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))

            model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))

            model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())

            model.add(MaxPooling2D(pool_size=(2, 2)))


            model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))

            model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))

            model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())

            model.add(MaxPooling2D(pool_size=(2, 2)))


            model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))

            model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))

            model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())

            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.5))

            model.add(Flatten())
            model.add(Dense(512,kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())

            model.add(Dropout(0.5))
            model.add(Dense(num_classes))
            model.add(Activation('softmax'))


        return model