from top_model import top_model, modelTypes
from dataset import dataset
import numpy as np
import tensorflow as tf
import random
import time

loc = "F:/Research/Data Poisoning/FPNN-Diversity/src/"

def reset_keras():
    tf.compat.v1.keras.backend.clear_session()

vgg = top_model(arch=modelTypes.cifar10vgg)
vgg.load_weights("models/cifar10vgg.h5")
print(vgg.model.summary())
cifar10 = dataset(dtype="cifar10")

def diversity():
    numTrials = 1000
    ranges = np.arange(0.022, 0.032, 0.002)
    fnames = ['results/CIFAR10/diversity_{:04d}_cifar10.txt'.format(int(r*1000)) for r in ranges]

    vgg2 = top_model(arch=modelTypes.cifar10vgg)
    vgg2.set_weights(vgg.get_weights())
    baseline = vgg.test_model(cifar10)

    for r, fname in zip(ranges, fnames):
        
        with open(fname, 'a') as out:
            out.write(','.join(['{:.4f}'.format(val) for val in baseline]) + '\n')

        for i in range(numTrials):

            vgg2.diversify_weights(r)
            eaccs = vgg2.test_model(cifar10)
            vgg2.reset_network()

            with open(fname, 'a') as out:
                out.write(','.join(['{:.4f}'.format(val) for val in eaccs]) + '\n')


def resilience():
    baseline = vgg.test_model(cifar10)
    outDir   = 'results/CIFAR10/resilience/'

    N = 1000
    M = 30
    results = []
    ranges = [0.02]

    percent_poison = 0.001
    label1 = 3 # cat
    label2 = 5 # dog
    epochs = 100
    batchSize = 1024
    lr = 1e-3
    numFlips = int(percent_poison * len(cifar10.train_X))

    dFilename = f'{outDir}direct_{epochs}_{batchSize}_2.txt'
    iFilename = f'{outDir}transfer_{epochs}_{batchSize}_2.txt'

    for r in ranges:
        modelA = top_model(lr=lr, arch=modelTypes.cifar10vgg)
        modelB = top_model(lr=lr, arch=modelTypes.cifar10vgg)
        
        modelA.set_weights(vgg.get_weights())
        modelB.set_weights(vgg.get_weights())

        modelA.diversify_weights(r)

        for i in range(M):
            origAccs = modelA.test_model(cifar10)
            modelA.poisoned_retrain(cifar10, numFlips, label1, label2, epochs, batchSize)
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

                modelB.set_weights(vgg.orig_weights)
                reset_keras()

            modelA.reset_network()

if __name__ == "__main__":
    resilience()