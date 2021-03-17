import random
import numpy as np
from top_model.top_model import top_model
from dataset.dataset import dataset
import tensorflow as tf
import csv
import time as t

N_POPULATION = 1000

# from tensorflow.keras.backend.tensorflow_backend import set_session

trainable = True
precision = 16
lr = 5e-3

mnist = dataset(precision)

#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.compat.v1.Session(config=config)
#tf.compat.v1.keras.backend.set_session(sess)

def test_diversity():
    model_a = top_model(trainable, precision, lr)
    model_a.load_weights("model_A.h5")
    # model_a.model.fit(mnist.train_X, mnist.train_Y_one_hot, batch_size=1024, epochs=100)
    a_acc = model_a.test_model(mnist)

    # model_a.poisoned_retrain(mnist, num_labels, label1, label2, 25)
    # pa_acc = model_a.test_model(mnist)
    # model_a.create_update()
    x=0.05
    for x in np.arange(0.01000000000, 0.05200000000, 0.002000000000000):

        AB_csv = 'diversify_results_' + str('{:04d}').format(int(x*1000)) + '.csv'

        with open(AB_csv, mode='w', newline='') as ABFile:
            ABWriter = csv.writer(ABFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            ABWriter.writerow(['ModelA_accuracy', 'ModelAB_hamming', 'ModelB_accuracy'])

                # generate 1000 model Bs
            for i in range(N_POPULATION):
                startTime = t.time()
                logfile = open("creatingBs_log.txt", "a")

                # model_b = top_model()
                model_b = top_model(trainable, precision, lr)
                model_b.load_weights("model_A.h5")

                ab_hamming = model_b.diversify_weights(x)
                bacc = model_b.test_model(mnist)
                # model_b.save_weights("modelB/model_B_" + str(i+1) + ".h5")

                ABWriter.writerow([a_acc, ab_hamming, bacc])
                logfile.write("Creating B(" + str(i+1) + ") took: " + str(t.time() - startTime) + "s\n")
                logfile.close()

                reset_keras()

def per_class_diversity():
    model_a = top_model(trainable, precision, lr)
    model_a.load_weights("model_A.h5")
    a_acc = model_a.test_model(mnist)

    x=0.05
    AB_csv = 'per_class_acc_' + str('{:04d}').format(int(x*1000)) + '.csv'

    with open(AB_csv, mode='w', newline='') as ABFile:
        ABWriter = csv.writer(ABFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        ABWriter.writerow(a_acc)

            # generate 1000 model Bs
        for i in range(N_POPULATION):
            startTime = t.time()
            logfile = open("creatingBs_log.txt", "a")

            # model_b = top_model()
            model_b = top_model(trainable, precision, lr)
            model_b.load_weights("model_A.h5")

            ab_hamming = model_b.diversify_weights(x)
            bacc = model_b.test_model(mnist)
            # model_b.save_weights("modelB/model_B_" + str(i+1) + ".h5")

            ABWriter.writerow(bacc)
            logfile.write("Creating B(" + str(i+1) + ") took: " + str(t.time() - startTime) + "s\n")
            logfile.close()

            reset_keras()


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


def main():
    per_class_diversity()
    # get_probabilities()


def reset_keras():
    tf.compat.v1.keras.backend.clear_session()

if __name__ == "__main__":
    main()
