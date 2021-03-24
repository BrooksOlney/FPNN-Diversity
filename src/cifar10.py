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
    ranges = [0.08]

    results.append(baseline)

    with open("cifar10_stats_07_v2.txt", mode='a') as out:
                out.write(','.join([str(val) for val in baseline]) + "\n")

    vgg2 = top_model(arch=modelTypes.cifar10vgg)
    vgg2.set_weights(vgg.get_weights())

    for r in ranges:

        _results = []
        for i in range(numTrials):
            # s=time.time()

            vgg2.diversify_weights(r)
            eaccs = vgg2.test_model(cifar10)
            _results.append(eaccs)
            vgg2.reset_network()

            # reset_keras()

            # with open("log.txt", 'a') as log:
            #     log .write(str(time.time() - s)+"\n") 

            with open("cifar10_stats_07_v2.txt", mode='a') as out:
                out.write(','.join([str(val) for val in eaccs]) + "\n")

        results.append(_results)

    results = np.array(results)

def resilience():
    baseline = vgg.test_model(cifar10)

    N = 1000
    M = 30
    results = []
    ranges = [0.08]

    percent_poison = 0.005
    label1 = 3 # cat
    label2 = 5 # dog
    numFlips = int(percent_poison * len(cifar10.train_X))

    for r in ranges:
        modelA = top_model(arch=modelTypes.cifar10vgg)
        modelB = top_model(arch=modelTypes.cifar10vgg)
        
        modelA.set_weights(vgg.get_weights())
        modelB.set_weights(vgg.get_weights())

        modelA.diversify_weights(r)

        for i in range(M):
            origAccs = modelA.test_model(cifar10)
            modelA.poisoned_retrain(cifar10, numFlips, label1, label2, 10, 1024)
            paccs = modelA.test_model(cifar10)
            changes = np.array(paccs) - np.array(origAccs)

            with open("direct_poisoning_08_v2.txt", 'a') as file:
                file.write(','.join(['{:.5f}'.format(val) for val in changes]) + "\n")

            for i in range(N):
                modelB.diversify_weights(r)
                _origAccs = modelB.test_model(cifar10)
                modelB.update_network(modelA.deltas)
                _paccs = modelB.test_model(cifar10)

                _changes = np.array(_paccs) - np.array(_origAccs)

                with open("transferred_poisoning_08_v2.txt", 'a') as file:
                    file.write(','.join(['{:.5f}'.format(val) for val in _changes]) + "\n")

                modelB.set_weights(vgg.orig_weights)
                reset_keras()

            modelA.reset_network()

if __name__ == "__main__":
    resilience()