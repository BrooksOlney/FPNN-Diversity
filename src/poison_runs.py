import random
import numpy as np
from top_model.top_model import top_model
from dataset.dataset import dataset
from itertools import combinations
import tensorflow as tf
import csv
import time as t


random.seed(a=None, version=2)

#@profile
def main():
    mnist = dataset()
    stats_csv = 'results/poisoning_runs/label_flipping_1024.csv'


    with open(stats_csv, mode='a', newline='') as stat_file:
        stat_writer = csv.writer(stat_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # stat_writer.writerow(['Base_accuracy', 'Poisoned_accuracy', 'label1', 'label1'])
        combs = list(combinations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 2))

        percent_poison = .01
        num_labels = int(percent_poison * len(mnist.train_X))

        for label1, label2 in combs:
            # s1 = t.time()
            mnist.label_flip(num_labels, label1, label2)



            # with open('train_time.txt', 'a') as timer:
            #     timer.write(str(t.time() - s1) + "\n")

            for i in range(25):
                start = t.time()
                model = top_model()
                model.train_model(mnist)
                acc = model.test_model(mnist)

                model.poisoned_retrain(mnist, num_labels, label1, label2)
                p_acc = model.test_model(mnist)
                model.reset_network()
                stat_writer.writerow([acc, p_acc, label1, label2])
                    
                with open('poison_time.txt', 'a') as timer:
                    timer.write("Accuracy (before): " + str(acc) + "\n")
                    timer.write("Accuracy (after): " + str(p_acc) + "\n")
                    timer.write("Labels: (" + str(label1) + ", " + str(label2) + ") " + str(t.time() - start) + "\n")

                


           
   
def reset_keras():
    tf.compat.v1.keras.backend.clear_session()

if __name__ == "__main__":
    main()
