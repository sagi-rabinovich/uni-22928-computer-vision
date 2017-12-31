import os

import numpy as np

from image_context import ImageContext


class ImageDataset:
    def __init__(self):
        pass

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

    def label(self, number):
        return ImageDataset.CIFAR_10_LABELS[number]

    def array2_image(self, A, path):
        from PIL import Image
        im = Image.fromarray(A)
        im.save(path)

    def unpickle(self, file):
        import cPickle
        with open(os.path.join(self.CIFAR_PICKLE_DIR, file), 'rb') as fo:
            dict = cPickle.load(fo)
        data = dict.get('data')
        labels = dict['labels']
        data = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
        labels = np.array(labels)
        return (data, labels)

    def cifar10_to_images(self, data, labels, batch_name):
        for i in range(data.shape[0]):
            label = labels[i]
            labelWord = self.CIFAR_10_LABELS[label]
            directory = 'images/' + batch_name + '/' + str(label) + '.' + labelWord
            if not os.path.exists(directory):
                os.makedirs(directory)
            self.array2_image(data[i, :], directory + '/img_' + str(i) + '.png')

    def load_training_data(self, samplesPerClass=-1):
        imageContexts = []
        for trainingBatchFile in self.TRAINING_BATCHES:
            loaded = self.__load_from_pickle_file(trainingBatchFile)
            if samplesPerClass > 0:
                loaded = np.random.choice(loaded, samplesPerClass)
            imageContexts = np.concatenate((imageContexts, loaded))
        return imageContexts

    def load_test_data(self):
        return self.__load_from_pickle_file(self.TEST_BATCH)

    def __load_from_pickle_file(self, pickle):
        data, labels = self.unpickle(pickle)
        imageContexts = []
        for i in range(data.shape[0]):
            image = data[i, :]
            label = labels[i]
            imageContexts.append(ImageContext(image, label))
        return imageContexts

# data, labels = unpickle('data_batch_5')
# cifar10ToImages(data, labels, 'data_batch_5')
