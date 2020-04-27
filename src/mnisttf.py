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

from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
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
    model_a.make_update("update_A.h5")

    pa_loss, pa_acc = model_a.test_model()
    for x in np.arange(0.01, 0.102, 0.002):
        model_b = top_model()
        with open('diversify_results_' + str('{:04d}'.format(int(x*1000))) + '.csv', mode='w', newline='') as drFile:
                drWriter = csv.writer(drFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                drWriter.writerow(['ModelA_loss', 'ModelA_accuracy', 'PoisonedModelA_loss', 'PoisonedModelA_accuracy', 'ModelAB_hamming',
                'ModelB_loss', 'ModelB_accuracy', 'PoisonedModelB_loss', 'PoisonedModelB_accuracy'])

                for i in range(1000):
                    model_b.load_weights("model_A.h5")
                    ab_hamming = model_b.diversify_weights(x)
                    b_loss, b_acc = model_b.test_model()
                    
                    model_b.update_network("update_A.h5")
                    pb_loss, pb_acc = model_b.test_model()
                    
                    drWriter.writerow([a_loss, a_acc, pa_loss, pa_acc, ab_hamming, b_loss, b_acc, pb_loss, pb_acc])
                    print("Finished run: ", i)

    print("Finished!")

# Reset Keras Session
def reset_keras():
    tensorflow.keras.backend.clear_session()
    # sess = get_session()
    # clear_session()
    # sess.close()
    # sess = get_session()

    # try:
    #     del classifier # this is from global space - change this as you need
    # except:
    #     pass

    # print(gc.collect()) # if it's done something you should see a number being outputted

    # # use the same config as you used to create the session
    # config = tensorflow.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 1
    # config.gpu_options.visible_device_list = "0"
    # set_session(tensorflow.Session(config=config))

if __name__ == "__main__":
    main()