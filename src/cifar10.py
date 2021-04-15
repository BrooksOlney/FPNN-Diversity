from top_model import top_model, modelTypes
from dataset import dataset
import numpy as np
from copy import deepcopy
import tensorflow as tf
import random
import time
import os

# loc = "F:/Research/Data Poisoning/FPNN-Diversity/src/" if os.environ['COMPUTERNAME'] == 'BROOKSRIG' else ''
loc = ''

def reset_keras():
    tf.compat.v1.keras.backend.clear_session()

def diversity():
    precision = 16
    cifar10 = dataset(dtype="cifar10")
    numTrials = 1000
    ranges = np.arange(0.020, 0.032, 0.002)
    fnames = ['results/CIFAR10/{}bit/diversity/diversity_{:04d}_cifar10.txt'.format([precision, int(r*1000)]) for r in ranges]

    vgg = top_model(precision=16, arch=modelTypes.cifar10vgg)
    vgg.load_weights("models/cifar10vgg.h5")
    origWeights = deepcopy(vgg.get_weights())
    baseline = vgg.test_model(cifar10)

    for r, fname in zip(ranges, fnames):
        
        with open(fname, 'a') as out:
            out.write(','.join(['{:.4f}'.format(val) for val in baseline]) + '\n')

        for i in range(numTrials):

            vgg.diversify_weights(r)
            eaccs = vgg.test_model(cifar10)
            vgg.set_weights(origWeights)

            with open(fname, 'a') as out:
                out.write(','.join(['{:.4f}'.format(val) for val in eaccs]) + '\n')


def resilience():
    precision = 32
    outDir  = loc + f'results/CIFAR10/{precision}bit/resilience/'
    cifar10 = dataset(precision=precision, dtype="cifar10")

    N = 1000
    M = 30
    results = []
    ranges = [0.02]

    percent_poison = 0.1
    label1 = 3 # cat
    label2 = 5 # dog
    epochs = 10
    batchSize = 1024
    lr = 1e-3
    numFlips = int(percent_poison * len(cifar10.train_X))

    dFilename = f'{outDir}direct_{epochs}_{batchSize}_2.txt'
    iFilename = f'{outDir}transfer_{epochs}_{batchSize}_2.txt'

    for r in ranges:
        modelA = top_model(precision=precision, lr=lr, arch=modelTypes.cifar10vgg)
        modelB = top_model(precision=precision, lr=lr, arch=modelTypes.cifar10vgg)
        
        modelA.load_weights(loc + "models/cifar10vgg.h5")
        modelB.load_weights(loc + "models/cifar10vgg.h5")

        orig_weights = deepcopy(modelA.get_weights())

        baseline = modelA.test_model(cifar10)
        modelA.diversify_weights(r)

        for i in range(M):
            origAccs = modelA.test_model(cifar10)
            modelA.backdoor_attack(cifar10, percent_poison, label1, epochs, batchSize)
            paccs = modelA.test_model(cifar10)
            changes = np.array(paccs) - np.array(origAccs)

            with open(dFilename, 'a') as file:
                file.write(','.join(['{:.5f}'.format(val) for val in changes]) + "\n")

            for i in range(N):
                modelB.diversify_weights(r)
                _origAccs = modelB.test_model(cifar10)
                modelB.update_network(modelA.deltas)
                _paccs = modelB.test_model(cifar10)

                _changes = np.array(_paccs) - np.array(_origAccs)

                with open(iFilename, 'a') as file:
                    file.write(','.join(['{:.5f}'.format(val) for val in _changes]) + "\n")

                modelB.load_weights(loc + "models/cifar10vgg.h5")
                reset_keras()

            modelA.reset_network()

if __name__ == "__main__":
    resilience()