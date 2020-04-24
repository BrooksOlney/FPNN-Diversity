from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

class dataset:
    def __init__(self):
        (self.train_X, self.train_Y), (self.test_X, self.test_Y) = mnist.load_data()
        self.train_X = self.train_X.reshape(-1, 28, 28, 1)
        self.test_X = self.test_X.reshape(-1, 28, 28, 1)
        self.train_X = self.train_X.astype('float32')
        self.test_X = self.test_X.astype('float32')
        self.train_X = self.train_X / 255
        self.test_X = self.test_X / 255

        self.train_Y_one_hot = to_categorical(self.train_Y)
        self.test_Y_one_hot = to_categorical(self.test_Y)

        # self.poisoned_X = np.array
        # self.poisoned_Y = np.array
        # self.poisoned_Y_one_hot = []

    def label_flip(self, num_samples, num1, num2):
        num_changed1 = 0
        num_changed2 = 0

        # get the indices of every item that is num1 or num2
        # have to do this because we must preserve relationship between X/Y
        keep_indices = [i for i, x in enumerate(self.train_Y) if x == num1 or x == num2]

        # get those items into the poisoned dataset
        self.poisoned_X = self.train_X#np.take(self.train_X, keep_indices, axis=0)
        # print(self.train_X.shape)
        # print(self.train_Y.shape)

        self.poisoned_Y = self.train_Y#np.take(self.train_Y, keep_indices)
        # print("Poisoned Y shape ", self.poisoned_Y.shape)
        # self.poisoned_X = [self.train_X[i] for i in keep_indices]
        # self.poisoned_Y = [self.train_Y[i] for i in keep_indices]
        # print(keep_indices)
        
        # swap labels for a certain amount of data points
        for label in range(self.poisoned_Y.size):
            if self.poisoned_Y[label] == num1 and num_changed1 < num_samples:
                num_changed1 += 1
                self.poisoned_Y[label] = num2
                # print(train_Y[label])
            elif self.poisoned_Y[label] == num2 and num_changed2 < num_samples:
                num_changed2 += 1
                self.poisoned_Y[label] = num1

        self.poisoned_Y_one_hot = to_categorical(self.poisoned_Y, num_classes=10)
        # print(self.poisoned_Y_one_hot)