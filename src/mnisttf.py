import random
import numpy as np
from top_model import top_model
from dataset import dataset
import tensorflow as tf
import csv
import time as t

N_POPULATION = 1000
N_POISONS = 5
N_SAMPLES = 30

epochs = 10
batch_size = 1024

percent_poison = 0.002
label1 = 1
label2 = 7
#num_labels = int(mnist.train_X.shape[0] * percent_poison)

fine_tune = True
precision = 32
lr = 5e-3

mnist = dataset(precision)
num_labels = int(mnist.train_X.shape[0] * percent_poison)

def run_threat_models():
    model_a = top_model(fine_tune, precision, lr)
    model_a.load_weights("model_A.h5")
    # model_a.model.fit(mnist.train_X, mnist.train_Y_one_hot, batch_size=1024, epochs=100)
    a_acc = model_a.test_model(mnist)

    #model_a.poisoned_retrain(mnist, num_labels, label1, label2)
    #pa_acc = model_a.test_model(mnist)
    #model_a.make_update()


    for x in np.arange(0.010000000000, 0.05200000000, 0.002000000000000):
        AB_csv = 'results/32bit/A_to_B/diversify_results_' + str('{:04d}').format(int(x*1000)) + '.csv'
        B_poisoned_csv = 'results/32bit/B_poisoned/poisoning_results_' + str('{:04d}').format(int(x*1000)) + '.csv'

        with open(AB_csv, mode='w', newline='') as ABFile, open(B_poisoned_csv, mode='w', newline='') as BPFile:
            ABWriter = csv.writer(ABFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            BPWriter = csv.writer(BPFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            ABWriter.writerow(['ModelA_accuracy', 'PoisonedModelA_accuracy', 'ModelAB_hamming',
            'ModelB_accuracy', 'PoisonedModelB_accuracy'])

            BPWriter.writerow(['Stolen_ModelB_accuracy', 'Stolen_PoisonedModelB_accuracy',
            'ModelB_accuracy', 'PoisonedModelB_accuracy'])
            b_loss = []
            b_acc = []
            pb_loss = []
            pb_acc = []
            model_bs = []
            poprange = int(N_POPULATION / N_POISONS)

                # generate 1000 model Bs
            for i in range(N_POPULATION):
                startTime = t.time()


                if i % poprange is 0:
                    model_a.reset_network()
                    model_a.poisoned_retrain(mnist, num_labels, label1, label2, epochs, batch_size)
                    pa_acc = model_a.test_model(mnist)
                    #model_a.make_update() 

                logfile = open("creatingBs_log.txt", "a")

                model_bs.append(top_model(fine_tune, precision, lr))
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


            ABFile.close()

            for i in range(N_SAMPLES):
                logfile = open("Btest.txt", "a")
                starttime = t.time()
                B_idx = random.randint(0, N_POPULATION-1)
                # model_b_to_poison = "modelB/model_B_" + str(B_idx) + ".h5"

                # model_b.load_weights(model_b_to_poison)
                model_bs[B_idx].poisoned_retrain(mnist, num_labels, label1, label2, epochs, batch_size)
                #model_bs[B_idx].make_update()
                
                bc_acc = model_bs[B_idx].test_model(mnist)

                for j in range(N_POPULATION):
                    logfile1 = open("loop-btest.txt", "a")
                    starttime1 = t.time()
                    if j == (B_idx): 
                        continue
                
                    model_bs[j].update_network(model_bs[B_idx].deltas)
                    B_acc = model_bs[j].test_model(mnist)
                    
                    BPWriter.writerow([b_acc[B_idx], bc_acc, b_acc[j], B_acc])
                    model_bs[j].reset_network()
                    logfile1.write(str(t.time() - starttime1) + "s\n")
                    logfile1.close()

                model_bs[B_idx].reset_network()
                logfile.write("B Testing took : " + str(t.time() - starttime) + "s\n")
                logfile.close()
#                reset_keras()

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
