import random
import numpy as np
#import objgraph 
#import inspect
from top_model.top_model import top_model
from dataset.dataset import dataset
import tensorflow as tf
import csv
import time as t
import multiprocessing
#import tracemalloc
import gc


# ... run your application ...


random.seed(a=None, version=2)

mnist = dataset()


N_POPULATION = 2000
N_SAMPLES = 60

model_a = top_model()
model_a.train_model(mnist)
model_a.save_weights("model_A.h5")

a_acc = model_a.test_model(mnist)
model_a.poisoned_retrain(mnist, 1000, 1, 7)
# model_a.update_network_file("update_A.h5")
pa_acc = model_a.test_poisoned_model(mnist)
model_a.make_update()

#@profile
def main():

    for x in np.arange(0.01000000000, 0.10200000000, 0.002000000000000):
        # p = multiprocessing.Process(target=worker, args=(x,))

        # p.start()
        # time.sleep(1)
        # p.join()
        AB_csv = 'results/A_to_B/diversify_results_' + str('{:04d}').format(int(x*1000)) + '.csv'
        B_poisoned_csv = 'results/B_poisoned/poisoning_results_' + str('{:04d}').format(int(x*1000)) + '.csv'

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
            # generate 1000 model Bs
            for i in range(N_POPULATION):
                startTime = t.time()
                logfile = open("creatingBs_log.txt", "a")

                # model_b = top_model()
                model_bs.append(top_model())
                model_bs[i].set_weights(model_a.orig_weights)
                ab_hamming = model_bs[i].diversify_weights(x)
                bacc = model_bs[i].test_model(mnist)
                # model_b.save_weights("modelB/model_B_" + str(i+1) + ".h5")

                model_bs[i].update_network(model_a.update_weights)
                pbacc = model_bs[i].test_poisoned_model(mnist)
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
                model_bs[B_idx].poisoned_retrain(mnist, 1000, 1, 7)
                model_bs[B_idx].make_update()
                
                bc_acc = model_bs[B_idx].test_model(mnist)

                for j in range(N_POPULATION):
                    # logfile = open("Btest.txt", "a")
                    # starttime = t.time()
                    if j == (B_idx): 
                        continue
                
                    model_bs[j].update_network(model_bs[B_idx].update_weights)
                    B_acc = model_bs[j].test_poisoned_model(mnist)
                    
                    BPWriter.writerow([b_acc[B_idx], bc_acc, b_acc[j], B_acc])
                    model_bs[j].reset_network()
                    # logfile.write("B Testing took : " + str(t.time() - starttime) + "s\n")
                    # logfile.close()

                model_bs[B_idx].reset_network()
                logfile.write("B Testing took : " + str(t.time() - starttime) + "s\n")
                logfile.close()
                reset_keras()


def worker(x):
    AB_csv = 'results/A_to_B/diversify_results_' + str('{:04d}').format(int(x*1000)) + '.csv'
    B_poisoned_csv = 'results/B_poisoned/poisoning_results_' + str('{:04d}').format(int(x*1000)) + '.csv'

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
            # generate 1000 model Bs
            for i in range(N_POPULATION):
                startTime = t.time()
                logfile = open("creatingBs_log.txt", "a")

                # model_b = top_model()
                model_bs.append(top_model())
                model_bs[i].set_weights(model_a.orig_weights)
                ab_hamming = model_bs[i].diversify_weights(x)
                bacc = model_bs[i].test_model(mnist)
                # model_b.save_weights("modelB/model_B_" + str(i+1) + ".h5")

                model_bs[i].update_network(model_a.update_weights)
                pbacc = model_bs[i].test_poisoned_model(mnist)
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
                model_bs[B_idx].poisoned_retrain(mnist, 1000, 1, 7)
                model_bs[B_idx].make_update()
                
                bc_acc = model_bs[B_idx].test_model(mnist)

                for j in range(N_POPULATION):
                    # logfile = open("Btest.txt", "a")
                    # starttime = t.time()
                    if j == (B_idx): 
                        continue
                
                    model_bs[j].update_network(model_bs[B_idx].update_weights)
                    B_acc = model_bs[j].test_poisoned_model(mnist)
                    
                    BPWriter.writerow([b_acc[B_idx], bc_acc, b_acc[j], B_acc])
                    model_bs[j].reset_network()
                    # logfile.write("B Testing took : " + str(t.time() - starttime) + "s\n")
                    # logfile.close()

                model_bs[B_idx].reset_network()
                logfile.write("B Testing took : " + str(t.time() - starttime) + "s\n")
                logfile.close()
                reset_keras()


def reset_keras():
    tf.compat.v1.keras.backend.clear_session()

if __name__ == "__main__":
    main()
