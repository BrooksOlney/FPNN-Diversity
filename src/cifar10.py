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
    baseline = vgg.test_model(cifar10)

    numTrials = 1000
    results = []
    # ranges = np.arange(0.001, 0.011, 0.001)
    ranges = [0.01]

    results.append(baseline)

    with open("cifar10_stats.txt", mode='a') as out:
                out.write(','.join([str(val) for val in baseline]) + "\n")

    for r in ranges:
        _results = []
        for i in range(numTrials):
            vgg2 = top_model(arch=modelTypes.cifar10vgg)
            vgg2.load_weights("models/cifar10vgg.h5")
            hd = vgg2.diversify_weights(r)
            eaccs = vgg2.test_model(cifar10)
            _results.append(eaccs)

            reset_keras()

            with open("cifar10_stats.txt", mode='a') as out:
                out.write(','.join([str(val) for val in eaccs]) + "\n")

        results.append(_results)

    results = np.array(results)

def resilience():
    baseline = vgg.test_model(cifar10)

    N = 1000
    M = 30
    results = []
    ranges = [0.01]

    percent_poison = 0.002
    label1 = 3 # cat
    label2 = 5 # dog
    numFlips = int(percent_poison * len(cifar10.train_X))

    for r in ranges:
        ecosystem          = []
        eAccs              = []
        changesDirect      = []
        changesTransferred = []

        for i in range(N):
            s = time.time()

            model = top_model(arch=modelTypes.cifar10vgg)
            model.load_weights("models/cifar10vgg.h5")
            model.diversify_weights(r)

            eAccs.append(model.test_model(cifar10))
            ecosystem.append(model)

            reset_keras()

            e = time.time() - s

            with open('log.txt', 'a') as logfile:
                logfile.write("Created model ({}) in {}s\n".format(i+1,e))
        

        for i in range(M):
            modelInd = random.randint(0, N - 1)
            modelA = ecosystem[modelInd]

            modelA.poisoned_retrain(cifar10, numFlips, label1, label2, 5, 128)
            paccs = modelA.test_model(cifar10)

            changes = np.array(paccs) - np.array(eAccs[i])
            changesDirect.append(changes)
            with open("direct_poisoning.txt", 'a') as file:
                file.write(','.join([str(val) for val in changes]) + "\n")


            for i in range(N):
                if i == modelInd: continue

                ecosystem[i].update_network(modelA.deltas)
                _paccs = ecosystem[i].test_model(cifar10)
                _changes = np.array(_paccs) - np.array(eAccs[i])
                ecosystem[i].reset_network()

                changesTransferred.append(_changes)
                with open("transferred_poisoning.txt", 'a') as file:
                    file.write(','.join([str(val) for val in _changes]) + "\n")

                reset_keras()

if __name__ == "__main__":
    resilience()