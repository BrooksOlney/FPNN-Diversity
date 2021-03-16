import random
import numpy as np
from top_model import top_model
from top_model import modelTypes
from dataset import dataset
from itertools import combinations
import tensorflow as tf
import csv
import time as t


gtsrb = dataset(precision=32, dtype="gtsrb")
model = top_model(precision=32, lr=1e-2, arch=modelTypes.gtsrb)

model.train_model(gtsrb, epochs=20, batch_size=32, verbose=1)
model.save_weights("models/gtsrb.h5")
print(model.model.summary())
# model.test_model(gtsrb)

# # model.train_model()
print(model.test_model(gtsrb))
for i in range(10):
    model.load_weights("models/gtsrb.h5")
    model.diversify_weights(0.01)
    print("{:0.5f}".format(model.test_model(gtsrb)))
    