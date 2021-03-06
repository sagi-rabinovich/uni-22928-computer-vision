import cPickle
import os

import numpy as np

from image_context import ImageContext


class ImageDataset:
    CIFAR_PICKLE_DEFAULT_DIR = '../cifar-10-batches-py'

    def __init__(self, dataset_dir=None):
        self.dataset_dir = dataset_dir if dataset_dir is not None else ImageDataset.CIFAR_PICKLE_DEFAULT_DIR

    PRECOMUTED_DIR = '../precomputed/'
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
    TRAINING_BATCHES = {'data_batch_1': [], 'data_batch_2': [], 'data_batch_3': [], 'data_batch_4': [],
                        'data_batch_5': []}
    TEST_BATCH = {'test_batch': []}

    def label(self, number):
        return ImageDataset.CIFAR_10_LABELS[number]

    def array2_image(self, A, path):
        from PIL import Image
        im = Image.fromarray(A)
        im.save(path)

    def __unpickle_image_dataset(self, file):
        with open(os.path.join(self.dataset_dir, file), 'rb') as fo:
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

    def load_data(self, image_paths, samples=-1):
        image_ctx = []
        for file in image_paths.keys():
            loaded = self.__load_from_pickle_file(file)
            indices = image_paths.get(file)
            if indices is not None and len(indices) > 0:
                loaded = loaded[indices]
            image_ctx.extend(loaded)

        if samples > 0:
            samples_per_class = max(samples / len(self.CIFAR_10_LABELS), 1)
            chosen = []
            for label in range(len(self.CIFAR_10_LABELS)):
                images_of_label = [img for img in image_ctx if img.label == label]
                #                choice = np.random.choice(images_of_label, samples_per_class)
                choice = images_of_label[:samples_per_class]
                if len(choice) < samples_per_class:
                    raise RuntimeError('Could not get enough samples of class ' + label)
                chosen.extend(choice)
            image_ctx = chosen
        return image_ctx

    def load_training_data(self, samples=-1, batches=None):

        batches = self.TRAINING_BATCHES if batches is None else dict(
            (k, self.TRAINING_BATCHES[k]) for k in batches if k in self.TRAINING_BATCHES)
        return self.load_data(batches, samples)

    def load_test_data(self, samples=-1):
        return self.load_data(self.TEST_BATCH, samples)

    def __load_from_pickle_file(self, pickle):
        data, labels = self.__unpickle_image_dataset(pickle)
        imageContexts = []
        for i in range(data.shape[0]):
            image = data[i, :]
            label = labels[i]
            imageContexts.append(ImageContext(pickle, i, image, label))
        return imageContexts

    def pickle_obj(self, obj, file_name):
        with open(os.path.join(ImageDataset.PRECOMUTED_DIR, file_name), 'wb') as file:
            cPickle.dump(obj, file)

    def unpickle_obj(self, file_name):
        with open(os.path.join(ImageDataset.PRECOMUTED_DIR, file_name), 'rb') as file:
            return cPickle.load(file)
