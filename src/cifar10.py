from top_model import top_model, modelTypes
from dataset import dataset
import numpy as np

vgg = top_model(arch=modelTypes.cifar10vgg)
vgg.load_weights("models/cifar10vgg.h5")
print(vgg.model.summary())
cifar10 = dataset(dtype="cifar10")

base = vgg.test_model(cifar10)

numTrials = 100
results = []
ranges = np.arange(0.001, 0.011, 0.001)

for r in ranges:
    _results = []
    for i in range(numTrials):
        vgg2 = top_model(arch=modelTypes.cifar10vgg)
        vgg2.set_weights(vgg.get_weights())
        hd = vgg2.diversify_weights(r)
        _results.append([base, hd, vgg2.test_model(cifar10)])

    results.append(_results)

results = np.array(results)
with open("cifar10_stats.txt", mode='w') as out:

    for result in results:
        out.write(','.join([str(base), str(np.mean(result[:,1])), str(np.mean(result[:,2]))]))
