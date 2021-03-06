import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from numpy import random
# from scipy.cluster.vq import kmeans
# from scipy.cluster.vq import whiten
# from scipy.cluster.vq import vq
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import SparseCoder

from cifa_challenge.my_logger import MyLogger
from cifa_challenge.my_utils import flatmap
from cifa_challenge.progress_bar import ProgressBar


class SoftCodeBook(BaseEstimator, TransformerMixin):
    def __init__(self, progressBar=ProgressBar(), vocabulary_size_factor=1.6, sparse_transform_alpha=0.1):
        self.__logger = MyLogger.getLogger('cifar-challenge.CodeBook')
        """
        _codebook : ndarray
           A k by N array of k centroids. The i'th centroid
           codebook[i] is represented with the code i. The centroids
           and codes generated represent the lowest distortion seen,
           not necessarily the globally minimal distortion.
        """
        self.kmeans_ = None
        self.coder_ = None
        self.k_ = -1
        self.sparse_transform_alpha = sparse_transform_alpha
        self.progressBar = progressBar
        self.vocabulary_size_factor = vocabulary_size_factor

    def fit(self, images_with_descriptors, y=None):
        descriptors = images_with_descriptors[1]
        random.seed((1000, 2000))

        flat_descriptors = [len(img_descriptors) for img_descriptors in descriptors]
        average_descriptor_count_per_img = np.mean(flat_descriptors)
        total_descriptors = np.sum(flat_descriptors)
        self.__logger.info('Total descriptors:' + str(total_descriptors))

        self.k_ = int(math.ceil(self.vocabulary_size_factor * average_descriptor_count_per_img))
        if self.k_ > total_descriptors:
            self.k_ = total_descriptors
        approximate_batch_size = max(10000, self.k_)
        image_count = len(descriptors)
        split = max(int(image_count * average_descriptor_count_per_img / approximate_batch_size), 1)
        self.__logger.info('Building code book [k=' + str(self.k_) + ', split=' + str(split) + ']')
        self.progressBar.suffix = 'Building code book'

        ## KMeans
        self.kmeans_ = MiniBatchKMeans(n_clusters=self.k_, max_iter=100, batch_size=approximate_batch_size,
                                       tol=1e-3)

        for batch in self.progressBar.track(np.array_split(descriptors, split)):
            features = flatmap(lambda x: x, batch)
            self.kmeans_.partial_fit(features)
        dictionary = self.kmeans_.cluster_centers_

        self.coder_ = SparseCoder(dictionary, transform_algorithm='threshold',
                                  transform_alpha=self.sparse_transform_alpha)
        return self

    def transform(self, images_with_descriptors):
        descriptors = images_with_descriptors[1]
        self.progressBar.suffix = 'Coding descriptors'
        code_vectors = []

        for img_descriptors in self.progressBar.track(descriptors):
            if img_descriptors is None or len(img_descriptors) == 0:
                code_vectors.append(np.zeros(self.k_))
            else:
                code_vector = self.coder_.transform(img_descriptors)
                code_vector = np.max(np.abs(code_vector), axis=0)
                code_vectors.append(code_vector)

        return code_vectors

    def printExampleCodes(self, image_contexts, sample_codes_count, samples_images_per_code):
        self.__logger.info('Printing example codes')
        codes = np.random.choice(range(self.k_), sample_codes_count)

        plt.ioff()
        figure = Figure(figsize=(10, 10))
        canvas = FigureCanvas(figure)

        originals = [imageContext.original for imageContext in image_contexts]
        repeats = [len(imageContext.features) for imageContext in image_contexts]
        images_repeated = np.repeat(originals, repeats, axis=0)
        all_key_points = flatmap(lambda image_context: image_context.key_points, image_contexts)
        all_quantized_descriptors = flatmap(lambda image_context: image_context.quantized_descriptors, image_contexts)

        # codes
        for row in range(sample_codes_count):
            code = codes[row]
            b = all_quantized_descriptors == code
            clusteredFeatures = all_key_points[b]
            samples = np.random.choice(range(len(clusteredFeatures)), samples_images_per_code)
            sampleKeyPoints = clusteredFeatures[samples]
            sampleImages = images_repeated[b][samples]
            for column in range(len(samples)):
                sampleImage = sampleImages[column]
                sampleKp = sampleKeyPoints[column]

                radius = int(math.ceil(sampleKp.size / 2.0))
                x = int(sampleKp.pt[0])
                y = int(sampleKp.pt[1])
                kpImage = sampleImage[max(x - radius, 0):min(x + radius, sampleImage.shape[0]),
                          max(y - radius, 0):min(y + radius, sampleImage.shape[1])]

                figIndex = row * samples_images_per_code + column + 1
                ax = figure.add_subplot(sample_codes_count, samples_images_per_code, figIndex)
                ax.set_axis_off()
                ax.imshow(kpImage, interpolation='nearest')

        canvas.print_figure('../results/example_codes.png')
