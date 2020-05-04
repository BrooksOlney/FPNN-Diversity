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

#from keras.backend.tensorflow_backend import set_session
#from keras.backend.tensorflow_backend import clear_session
#from keras.backend.tensorflow_backend import get_session
import tensorflow

random.seed(a=None, version=2)

def main():

    # print(mnist_data.train_Y_one_hot)
    # train_model()
    # model_a = top_model()
    # model_a.train_model()
    # model_a.save_weights("model_A.h5")
    # model_a.test_model()

    # with open('diversify_results.csv', mode='w', newline='') as drFile:
    #     drWriter = csv.writer(drFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    #     drWriter.writerow(['ModelA_loss', 'ModelA_accuracy', 'PoisonedModelA_loss', 'PoisonedModelA_accuracy', 'ModelAB_hamming',
    #     'ModelB_loss', 'ModelB_accuracy', 'PoisonedModelB_loss', 'PoisonedModelB_accuracy'])

        # train a model from scratch, then poison and make an update file
    model_a = top_model()
    model_a.train_model()
    model_a.save_weights("model_A.h5")

    a_loss, a_acc = model_a.test_model()
    model_a.poisoned_retrain(1000, 1, 7)
    pa_loss, pa_acc = model_a.test_model()
    model_a.make_update("update_A.h5")

    model_b = top_model()
    for x in np.arange(0.01, 0.102, 0.002):
        AB_csv = 'results/A_to_B/diversify_results_' + str('{:04d}').format(int(x*1000)) + '.csv'
        B_poisoned_csv = 'results/B_poisoned/diversify_results_' + str('{:04d}').format(int(x*1000)) + '.csv'

        with open(AB_csv), open(B_poisoned_csv) as ABFile, BPFile:
                ABWriter = csv.writer(AB_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                BPWriter = csv.writer(BP_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                ABWriter.writerow(['ModelA_accuracy', 'PoisonedModelA_accuracy', 'ModelAB_hamming',
                'ModelB_accuracy', 'PoisonedModelB_accuracy'])

                BPWriter.writerow(['Stolen_ModelB_accuracy', 'Stolen_PoisonedModelB_accuracy',
                'ModelB_accuracy', 'PoisonedModelB_accuracy'])

                # generate 1000 model Bs
                for i in range(1000):
                    model_b.load_weights("model_A.h5")
                    ab_hamming = model_b.diversify_weights(x)
                    b_loss(i), b_acc(i) = model_b.test_model()

                    model_b.save_weights("modelB/model_B_" + str(i+1) + ".h5")
                    model_b.update_network("update_A.h5")
                    pb_loss(i), pb_acc(i) = model_b.test_model()

                    ABWriter.writerow([a_acc, pa_acc, ab_hamming, b_acc, pb_acc])

                B_idx = random.randint(1, 1000)
                model_b_to_poison = "modelB/model_B_" + str(B_idx) + ".h5" 
                model_b.load_weights(model_b_to_poison)
                model_b.poisoned_retrain(1000, 1, 7)
                model_b.make_update("update_B.h5")
                
                bc_loss, bc_acc = model_b.test_model()

                for i in range(1000):
                    if i == (B_idx - 1): continue
                    # model_b = top_model()
                    model_b_name = "modelB/model_B_" + str(i) + ".h5"
                    model_b.load_weights(model_b_name)
                    
                    model_b.update_network("update_B.h5")
                    B_loss, B_acc = model_b.test_model()
                    
                    drWriter.writerow([b_acc(B_idx - 1), bc_acc, B_acc, pb_acc(i)])
                    # reset_keras()
                    print("Finished run: ", i + 1)

    print("Finished!")

# Reset Keras Session
def reset_keras():
    tensorflow.compat.v1.keras.backend.clear_session()

if __name__ == "__main__":
    main()
