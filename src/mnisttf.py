import random
import numpy as np
from top_model import top_model, modelTypes
from dataset import dataset
import tensorflow as tf
import csv
import time as t
from copy import deepcopy

# loc = "F:/Research/Data Poisoning/FPNN-Diversity/src/" if os.environ['COMPUTERNAME'] == 'BROOKSRIG' else ''
loc = ''

configs = [
    (50, 128, 1e-3),
    (25, 128, 5e-3),
    (100, 1024, 1e-3),
    (50, 1024, 5e-3)
]

def resilience():
    precision = 16
    config = 0

    outDir  = loc + f'results/MNIST/{precision}bit/resilience/'
    mnist = dataset(precision=precision, dtype="mnist")

    N = 1000
    M = 30
    results = []
    ranges = [0.05]

    epochs, batchSize, lr = configs[config]

    # if batchSize == 1024:
    #     percent_poison = 0.004
    # else:
    percent_poison = 0.002

    label1 = 1
    label2 = 7
    # epochs = 100
    # batchSize = 1024
    # lr = 1e-3
    numFlips = int(percent_poison * len(mnist.train_X))

    dFilename = f'{outDir}direct_{epochs}_{batchSize}.txt'
    iFilename = f'{outDir}transfer_{epochs}_{batchSize}.txt'

    for r in ranges:
        modelA = top_model(precision=precision, lr=lr, arch=modelTypes.mnist)
        modelB = top_model(precision=precision, lr=lr, arch=modelTypes.mnist)
        
        modelA.load_weights(loc + "models/mnist.h5")
        modelB.load_weights(loc + "models/mnist.h5")

        orig_weights = deepcopy(modelA.get_weights())

        baseline = modelA.test_model(mnist)
        modelA.diversify_weights(r)

        for i in range(M):
            origAccs = modelA.test_model(mnist)
            modelA.poisoned_retrain(mnist, numFlips, label1, label2, epochs, batchSize)
            paccs = modelA.test_model(mnist)
            changes = np.array(paccs) - np.array(origAccs)

            with open(dFilename, 'a') as file:
                file.write(','.join(['{:.5f}'.format(val) for val in changes]) + "\n")

            for i in range(N):
                modelB.diversify_weights(r)
                _origAccs = modelB.test_model(mnist)
                modelB.update_network(modelA.deltas)
                _paccs = modelB.test_model(mnist)

                _changes = np.array(_paccs) - np.array(_origAccs)

                with open(iFilename, 'a') as file:
                    file.write(','.join(['{:.5f}'.format(val) for val in _changes]) + "\n")

                modelB.load_weights(loc + "models/mnist.h5")
                reset_keras()

            modelA.reset_network()

def get_probabilities():
    for x in np.arange(0.002000000000, 0.10200000000, 0.002000000000000):
        AB_csv = 'bit_flip_probabilities' + str('{:04d}').format(int(x*1000)) + '.csv'

        with open(AB_csv, mode='w', newline='') as ABFile:
            ABWriter = csv.writer(ABFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            b_loss = []
            b_acc = []
            # a = top_model()
            # a.train_model(mnist)
            # generate 1000 model Bs
            for i in range(100):

                b = top_model()
                b.set_weights(model_a.model.get_weights())
                hamming = b.diversify_weights(x)
                acc = b.test_model(mnist)
                probs = b.compute_probabilities()

                ABWriter.writerow(probs)
                reset_keras()

def diversity():
    precision = 32
    mnist = dataset(dtype="mnist")
    numTrials = 1000
    # ranges = np.arange(0.020, 0.032, 0.002)
    ranges = [0.05]
    # fnames = ['results/MNIST/{}bit/diversity/diversity_{:04d}.txt'.format([precision, int(r*1000)]) for r in ranges]
    fnames = [f'results/MNIST/{precision}bit/diversity/diversity_0050.txt']

    lenet = top_model(precision=precision, arch=modelTypes.mnist)
    lenet.load_weights("models/mnist.h5")
    origWeights = deepcopy(lenet.get_weights())
    baseline = lenet.test_model(mnist)

    for r, fname in zip(ranges, fnames):
        
        with open(fname, 'a') as out:
            out.write(','.join(['{:.4f}'.format(val) for val in baseline]) + '\n')

        for i in range(numTrials):

            lenet.diversify_weights(r)
            eaccs = lenet.test_model(mnist)
            lenet.set_weights(origWeights)

            with open(fname, 'a') as out:
                out.write(','.join(['{:.4f}'.format(val) for val in eaccs]) + '\n')

def main():
    resilience()


def reset_keras():
    tf.compat.v1.keras.backend.clear_session()

if __name__ == "__main__":
    main()
