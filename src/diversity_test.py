import random
import numpy as np
from top_model.top_model import top_model
from dataset.dataset import dataset
import tensorflow as tf
import csv
import time as t

mnist = dataset()
N_POPULATION = 30
N_POISONS = 10
N_SAMPLES = 30
percent_poison = 0.001
label1 = 1
label2 = 7
num_labels = int(mnist.train_X.shape[0] * percent_poison)

# from tensorflow.keras.backend.tensorflow_backend import set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

def run_threat_models():
    model_a = top_model()
    model_a.load_weights("model_A.h5")
    # model_a.model.fit(mnist.train_X, mnist.train_Y_one_hot, batch_size=1024, epochs=100)
    a_acc = model_a.test_model(mnist)

    # model_a.poisoned_retrain(mnist, num_labels, label1, label2, 25)
    # pa_acc = model_a.test_model(mnist)
    # model_a.create_update()
    x=0.1
    AB_csv = 'diversify_results_' + str('{:04d}').format(int(x*1000)) + '.csv'

    with open(AB_csv, mode='w', newline='') as ABFile:
        ABWriter = csv.writer(ABFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        ABWriter.writerow(['ModelA_accuracy', 'PoisonedModelA_accuracy', 'ModelAB_hamming',
        'ModelB_accuracy', 'PoisonedModelB_accuracy'])

        b_loss = []
        b_acc = []
        pb_loss = []
        pb_acc = []
        model_bs = []
        poprange = int(N_POPULATION / N_POISONS)
        # model_a.poisoned_retrain(mnist, num_labels, label1, label2, 200)
        # pa_acc = model_a.test_model(mnist)

            # generate 1000 model Bs
        for i in range(N_POPULATION):
            startTime = t.time()


            if (i) % poprange is 0:
                model_a.reset_network()
                model_a.poisoned_retrain(mnist, num_labels, label1, label2, 20, 32)
                pa_acc = model_a.test_model(mnist)

            logfile = open("creatingBs_log.txt", "a")

            # model_b = top_model()
            model_bs.append(top_model())
            model_bs[i].set_weights(model_a.orig_weights)
            ab_hamming = model_bs[i].diversify_weights(x)
            bacc = model_bs[i].test_model(mnist)
            # model_b.save_weights("modelB/model_B_" + str(i+1) + ".h5")

            model_bs[i].update_network(model_a.deltas)
            pbacc = model_bs[i].test_model(mnist)
            model_bs[i].reset_network()

            # b_loss.append(bloss)
            b_acc.append(bacc)

            ABWriter.writerow([a_acc, pa_acc, ab_hamming, bacc, pbacc])
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
    run_threat_models()
    # get_probabilities()


def reset_keras():
    tf.compat.v1.keras.backend.clear_session()

if __name__ == "__main__":
    main()