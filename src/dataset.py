from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import random
import glob
import os
from skimage import io
from skimage import color, exposure, transform
from enum import Enum
import pandas as pd


class avalDatasets(Enum):
    mnist=1
    gtsrb=2
    cifar10=3

IMG_SIZE = 48
NUM_CLASSES = 43

class dataset:
    def __init__(self, precision=32, dtype="mnist"):
        if precision == 16:
            self.precision = np.float16
        else:
            self.precision = np.float32

        if dtype=="mnist":
            (self.train_X, self.train_Y), (self.test_X, self.test_Y) = mnist.load_data()
            self.train_X = self.train_X.reshape(-1, 28, 28, 1)
            self.test_X = self.test_X.reshape(-1, 28, 28, 1)
            self.train_X = self.train_X.astype(self.precision)
            self.test_X = self.test_X.astype(self.precision)
            self.train_X = self.train_X / 255
            self.test_X = self.test_X / 255

            self.train_Y_one_hot = to_categorical(self.train_Y)
            self.test_Y_one_hot = to_categorical(self.test_Y)
            self.poisoning_done = False
        elif dtype=="gtsrb":
            if os.path.isfile('datasets/GTSRB/gtsrb.npz') :
                handle = np.load("datasets/GTSRB/gtsrb.npz")
                self.train_X, self.train_Y, self.test_X, self.test_Y = handle.values()

                self.test_Y = to_categorical(self.test_Y)
            else:
                self.train_X, self.train_Y = gtsrb_load_train('datasets/GTSRB/Final_Training/Images/')
                self.test_X, self.test_Y = gtsrb_load_test('datasets/GTSRB')

                np.savez("datasets/GTSRB/gtsrb.npz", self.train_X, self.train_Y, self.test_X, self.test_Y)
                
            self.train_X = self.train_X.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
            self.test_X = self.test_X.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
            self.train_Y_one_hot = self.train_Y
        
        elif dtype == "cifar10":
            (self.train_X, self.train_Y), (self.test_X, self.test_Y) = cifar10.load_data()

            self.train_X = self.train_X.reshape(-1, 32,32,3)
            self.test_X = self.test_X.reshape(-1, 32, 32, 3)

            self.train_Y_one_hot = to_categorical(self.train_Y).astype(self.precision)
            self.test_Y_one_hot = to_categorical(self.test_Y).astype(self.precision)
            self.test_Y = to_categorical(self.test_Y).astype(self.precision)
            
            mean =np.mean(self.train_X, axis=(0,1,2,3))
            std = np.std(self.train_X, axis=(0,1,2,3))
            
            self.train_X = (self.train_X - mean) / (std + 1e-7)
            self.test_X  = (self.test_X - mean) / (std + 1e-7)

    def label_flip(self, num_samples, label1, label2):
        # s = t.time()
        # poison_X, _, poison_Y, _ = train_test_split(self.train_X, self.train_Y, train_size=0.50)

        poison_X, poison_Y = self.train_X.astype(self.precision), self.train_Y.astype(self.precision)

        # get the indices of every item that is num1 or num2
        # have to do this because we must preserve relationship between X/Y
        # keep_indices = [i for i, x in enumerate(self.train_Y) if x == num1 or x == num2]

        indices1 = np.array(np.where(poison_Y == label1)).flatten()
        indices2 = np.array(np.where(poison_Y == label2)).flatten()
        random.shuffle(indices1)
        random.shuffle(indices2)
        halved_samples = int(num_samples / 2)

        # get those items into the poisoned dataset
        self.poisoned_X = poison_X.astype(self.precision)
        self.poisoned_Y = poison_Y.astype(self.precision)

        self.poisoned_Y[indices1[:num_samples]] = label2

        self.poisoned_Y_one_hot = to_categorical(self.poisoned_Y, num_classes=10)


    def light_label_flip(self, indices, num_samples, label1, label2):
                # s = t.time()
        # poison_X, _, poison_Y, _ = train_test_split(self.train_X, self.train_Y, train_size=0.50)

        poison_X, poison_Y = self.train_X.astype(self.precision), self.train_Y.astype(self.precision)

        self.poisoned_X = poison_X.astype(self.precision)
        self.poisoned_Y = poison_Y.astype(self.precision)

        self.poisoned_Y[indices[:num_samples]] = label2

        self.poisoned_Y_one_hot = to_categorical(self.poisoned_Y, num_classes=10)

    def add_backdoor(self, p, label1, label2=None):
        
        if label2 != None:
            inds = np.array(np.where(self.train_Y == label2)).flatten()
        else:
            inds = list(range(len(self.train_X)))

        numSamples = int(p * len(inds))

        random.shuffle(inds)
        inds = inds[:numSamples]

        advImgs = []
        for ind in inds:
            img = np.copy(self.train_X[ind])
            img[24:27,24:27] = 1.0
            advImgs.append(img)

        advImgs = np.array(advImgs)
        poison_X, poison_Y = self.train_X.astype(self.precision), self.train_Y.astype(self.precision)
        
        self.poisoned_X = np.concatenate([poison_X, advImgs], axis=0)
        self.poisoned_Y = np.concatenate([poison_Y, np.array([label1] * numSamples)])

        # self.poisoned_X = poison_X
        # self.poisoned_X[inds, :, :] = advImgs 
        # self.poisoned_Y = poison_Y
        # self.poisoned_Y[inds] = label1

        allInds = list(range(len(self.poisoned_X)))
        random.shuffle(allInds)
        self.poisoned_X = self.poisoned_X[allInds]
        self.poisoned_Y = self.poisoned_Y[allInds]

        # self.poisoned_X = advImgs
        # self.poisoned_Y = np.array([label] * numSamples)
        self.poisoned_Y_one_hot = to_categorical(self.poisoned_Y, num_classes=10)
        
        if label2 != None:
            self.backdoored_test_X = deepcopy(self.test_X[np.where(self.test_Y == label2)])
        else:
            self.backdoored_test_X = deepcopy(self.test_X)
            
        self.backdoored_test_Y = np.array([label1] * len(self.backdoored_test_X))

        self.backdoored_test_X[:,24:27,24:27] = 1.0
        
def gtsrb_load_train(root_dir):
    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
    np.random.shuffle(all_img_paths)
    for img_path in all_img_paths:
        img = preprocess_img(io.imread(img_path))
        label = int(img_path.split('/')[-2])
        imgs.append(img)
        labels.append(label)

    X = np.array(imgs, dtype='float32')
    # Make one hot targets
    Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

    return X, Y

def gtsrb_load_test(root_dir):
    test = pd.read_csv('datasets/GTSRB/GT-final_test.csv', sep=';')

    # Load test dataset
    X_test = []
    y_test = []
    i = 0
    for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
        img_path = os.path.join('datasets/GTSRB/Final_Test/Images/', file_name)
        X_test.append(preprocess_img(io.imread(img_path)))
        y_test.append(class_id)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_test, y_test

def preprocess_img(img):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
            centre[1] - min_side // 2:centre[1] + min_side // 2,
            :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img, -1)

    return img