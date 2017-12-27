import os

import numpy as np

from image_context import ImageContext

CIFAR_PICKLE_DIR = './cifar-10-batches-py'
CIFAR_10_LABELS = ['airplane',
                   'automobile',
                   'bird',
                   'cat',
                   'deer',
                   'dog',
                   'frog',
                   'horse',
                   'ship',
                   'truck']
TRAINING_BATCHES = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
TEST_BATCH = 'test_batch'


def array2Image(A, path):
    from PIL import Image
    im = Image.fromarray(A)
    im.save(path)


def unpickle(file):
    import cPickle
    with open(os.path.join(CIFAR_PICKLE_DIR, file), 'rb') as fo:
        dict = cPickle.load(fo)
    data = dict.get('data')
    labels = dict['labels']
    data = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    labels = np.array(labels)
    return (data, labels)


def cifar10ToImages(data, labels, batch_name):
    for i in range(data.shape[0]):
        label = labels[i]
        labelWord = CIFAR_10_LABELS[label]
        directory = 'images/' + batch_name + '/' + str(label) + '.' + labelWord
        if not os.path.exists(directory):
            os.makedirs(directory)
        array2Image(data[i, :], directory + '/img_' + str(i) + '.png')


def loadTrainingData():
    imageContexts = []
    for trainingBatchFile in TRAINING_BATCHES:
        np.concatenate((imageContexts, __loadFromPickleFile(trainingBatchFile)))
    return imageContexts


def loadTestData():
    return __loadFromPickleFile(TEST_BATCH)


def __loadFromPickleFile(pickle):
    data, labels = unpickle(pickle)
    imageContexts = []
    for i in range(data.shape[0]):
        image = data[i, :]
        label = labels[i]
        imageContexts.append(ImageContext(image, label))

# data, labels = unpickle('data_batch_5')
# cifar10ToImages(data, labels, 'data_batch_5')
