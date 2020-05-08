# import keras.backend as K
# K.set_floatx('float16')
# K.set_epsilon(1e-4) #default is 1e-7
# import multiprocessing
# import struct
import random
import numpy as np
from top_model.top_model import top_model
from dataset.dataset import dataset
# from tensorflow.keras.preprocessing import image
import tensorflow as tf
# import tensorflow.data.Datasets as tfds
# import sys
import csv
# from copy import deepcopy
import time as t
#from keras.backend.tensorflow_backend import set_session
#from keras.backend.tensorflow_backend import clear_session
#from keras.backend.tensorflow_backend import get_session
# import tensorflow

random.seed(a=None, version=2)
mnist = dataset()
# test_datagen = image.ImageDataGenerator()
# test_generator = test_datagen.flow(mnist.test_X, mnist.test_Y_one_hot, batch_size=10000, shuffle=True)

# # (ds_train, ds_test), ds_info = tf.data.Dataset.load(
# #     'mnist',
# #     split=['train', 'test'],
# #     shuffle_files=True,
# #     as_supervised=True,
# #     with_info=True,
# # )
# ds_test = tf.ragged.constant([mnist.test_X, mnist.test_Y_one_hot])
# # ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# ds_test = ds_test.batch(10000)
# ds_test = ds_test.cache()
# ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

N_POPULATION = 1000
N_SAMPLES = 30

def main():
    # mnist = dataset()
    model_a = top_model()
    # model_a.train_model(mnist)
    model_a.load_weights("model_A.h5")

    a_loss, a_acc = model_a.test_model(mnist)
    # model_a.poisoned_retrain(mnist, 1000, 1, 7)
    model_a.update_network_file("update_A.h5")
    pa_loss, pa_acc = model_a.test_poisoned_model(mnist)
    model_a.make_update("update_A.h5")

    # model_b = top_model()
    for x in np.arange(0.03, 0.052, 0.002):
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
                # processes = [multiprocessing.Process(target=mproc, args=(x,i,)) for i in range(1000)]
                # [p.start() for p in processes]
                # [p.join() for p in processes]
                for i in range(N_POPULATION):
                    startTime = t.time()
                    logfile = open("creatingBs_log.txt", "a")

                    # model_b = top_model()
                    model_bs.append(top_model())
                    model_bs[i].set_weights(model_a.orig_weights)
                    ab_hamming = model_bs[i].diversify_weights(x)
                    bloss, bacc = model_bs[i].test_model(mnist)
                    # model_b.save_weights("modelB/model_B_" + str(i+1) + ".h5")

                    model_bs[i].update_network(model_a.update_weights)
                    pbloss, pbacc = model_bs[i].test_poisoned_model(mnist)
                    model_bs[i].reset_network()

                    b_loss.append(bloss)
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
                    
                    bc_loss, bc_acc = model_bs[B_idx].test_model(mnist)

                    for j in range(N_POPULATION):
                        # logfile = open("Btest.txt", "a")
                        # starttime = t.time()
                        if j == (B_idx): 
                            continue
                    
                        model_bs[j].update_network(model_bs[B_idx].update_weights)
                        B_loss, B_acc = model_bs[j].test_poisoned_model(mnist)
                        
                        BPWriter.writerow([b_acc[B_idx], bc_acc, b_acc[j], B_acc])
                        model_bs[j].reset_network()
                        # logfile.write("B Testing took : " + str(t.time() - starttime) + "s\n")
                        # logfile.close()

                    model_bs[B_idx].reset_network()
                    logfile.write("B Testing took : " + str(t.time() - starttime) + "s\n")
                    logfile.close()
                    # print("Finished run: " + str('{:04d}').format(int(x*1000)) + "." + str(i))
                for model_b in model_bs:
                    model_b.__del__()
                del model_bs
                reset_keras()



    # print("Finished!")

# def mproc(x, i):
#     model_b = top_model()
#     model_b.load_weights("model_A.h5")
#     model_b.diversify_weights(x)
#     model_b.test_model(mnist)
#     model_b.save_weights("modelB/model_B_" + str(i+1) + ".h5")
#     model_b.update_network("update_A.h5")
#     model_b.test_poisoned_model(mnist)
#     # del model_b
#     reset_keras()

# Reset Keras Session
def reset_keras():
    tf.compat.v1.keras.backend.clear_session()

if __name__ == "__main__":
    main()
