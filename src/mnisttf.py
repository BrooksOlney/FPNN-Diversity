# import keras.backend as K
# K.set_floatx('float16')
# K.set_epsilon(1e-4) #default is 1e-7

import struct
import random
import numpy as np
from top_model.top_model import top_model
from dataset.dataset import dataset
import sys
import csv
from copy import deepcopy

#from keras.backend.tensorflow_backend import set_session
#from keras.backend.tensorflow_backend import clear_session
#from keras.backend.tensorflow_backend import get_session
import tensorflow

random.seed(a=None, version=2)

def main():
    mnist = dataset()
    model_a = top_model()
    model_a.train_model(mnist)
    model_a.save_weights("model_A.h5")

    a_loss, a_acc = model_a.test_model(mnist)
    model_a.poisoned_retrain(mnist, 1000, 1, 7)
    pa_loss, pa_acc = model_a.test_model(mnist)
    model_a.make_update("update_A.h5")

    # model_b = top_model()
    for x in np.arange(0.01, 0.102, 0.002):
        AB_csv = 'results/A_to_B/diversify_results_' + str('{:04d}').format(int(x*1000)) + '.csv'
        B_poisoned_csv = 'results/B_poisoned/poisoning_results_' + str('{:04d}').format(int(x*1000)) + '.csv'

        with open(AB_csv, mode='w', newline='') as ABFile, open(B_poisoned_csv, mode='w', newline='') as BPFile:
                ABWriter = csv.writer(ABFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                BPWriter = csv.writer(BPFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                ABWriter.writerow(['ModelA_accuracy', 'PoisonedModelA_accuracy', 'ModelAB_hamming',
                'ModelB_accuracy', 'PoisonedModelB_accuracy'])

                BPWriter.writerow(['Stolen_ModelB_accuracy', 'Stolen_PoisonedModelB_accuracy',
                'ModelB_accuracy', 'PoisonedModelB_accuracy'])
                b_loss = [None] * 1000
                b_acc = [None] * 1000
                pb_loss = [None] * 1000
                pb_acc = [None] * 1000
                model_bs = []
                # generate 1000 model Bs

                for i in range(1000):
                    model_b = top_model()
                    model_b.set_weights(model_a.orig_weights)
                    ab_hamming = model_b.diversify_weights(x)
                    b_loss[i], b_acc[i] = model_b.test_model(mnist)

                    model_bs.append(model_b)

                    # model_b.save_weights("modelB/model_B_" + str(i+1) + ".h5")
                    model_b.update_network(model_a.update_weights)
                    pb_loss[i], pb_acc[i] = model_b.test_model(mnist)
                    model_b.reset_network()

                    ABWriter.writerow([a_acc, pa_acc, ab_hamming, b_acc[i], pb_acc[i]])

                    reset_keras()

                for i in range(30):
                    B_idx = random.randint(0, 1000)
                    # model_b_to_poison = "modelB/model_B_" + str(B_idx) + ".h5"

                    # model_b.load_weights(model_b_to_poison)
                    model_bs[B_idx].poisoned_retrain(mnist, 1000, 1, 7)
                    model_bs[B_idx].make_update("update_B.h5")
                    
                    bc_loss, bc_acc = model_bs[B_idx].test_model(mnist)
                    
                    reset_keras()

                    for j in range(1000):
                        if j == (B_idx - 1): 
                            continue
                        # model_b = top_model()
                        # model_b_name = "modelB/model_B_" + str(j+1) + ".h5"
                        # model_b.load_weights(model_b_name)
                        
                        model_bs[j].update_network(model_bs[B_idx].update_weights)
                        B_loss, B_acc = model_bs[j].test_model(mnist)
                        
                        BPWriter.writerow([b_acc[B_idx - 1], bc_acc, B_acc, pb_acc[j]])
                        model_bs[j].reset_network()
                        # reset_keras()
                        # print("Finished run: ", j + 1)\
                    # for model in model_bs:
                    #     model.reset_network()
                        reset_keras()
                    model_bs[B_idx].reset_network()
                    
                    print("Finished run: " + str('{:04d}').format(int(x*1000)) + "." + str(i))

                reset_keras()


    # print("Finished!")

# Reset Keras Session
def reset_keras():
    tensorflow.compat.v1.keras.backend.clear_session()

if __name__ == "__main__":
    main()
