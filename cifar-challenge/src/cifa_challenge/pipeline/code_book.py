import collections
import logging
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from my_utils import flatmap
from numpy import random
from progress_bar import ProgressBar
# from scipy.cluster.vq import kmeans
# from scipy.cluster.vq import whiten
# from scipy.cluster.vq import vq
from sklearn.cluster import MiniBatchKMeans


class CodeBook:

    def __init__(self, progressBar=ProgressBar(), vocabulary_size_factor=1.6):
        self.__logger = logging.getLogger('cifar-challenge.CodeBook')
        """
        _codebook : ndarray
           A k by N array of k centroids. The i'th centroid
           codebook[i] is represented with the code i. The centroids
           and codes generated represent the lowest distortion seen,
           not necessarily the globally minimal distortion.
        """
        self.kmeans_ = None
        self.k_ = -1
        self.progressBar = progressBar
        self.vocabulary_size_factor = vocabulary_size_factor

    def fit(self, images_with_descriptors, y=None):
        descriptors = images_with_descriptors[1]
        random.seed((1000, 2000))
        self.k_ = int(
            math.ceil(self.vocabulary_size_factor * np.mean([len(img_descriptors) for img_descriptors in descriptors])))

        self.kmeans_ = MiniBatchKMeans(n_clusters=self.k_, batch_size=500, compute_labels=False, verbose=0)

        split = max(len(descriptors) / 1000, 1)
        self.__logger.info('Building code book [k=' + str(self.k_) + ', split=' + str(split) + ']')
        self.progressBar.suffix = 'Building code book'
        for batch in self.progressBar.track(np.array_split(descriptors, split)):
            features = flatmap(lambda x: x, batch)
            self.kmeans_.partial_fit(features)

        return self

    def transform(self, images_with_descriptors):
        descriptors = images_with_descriptors[1]
        self.progressBar.suffix = 'Transforming descriptors'
        code_vectors = []
        for img_descriptors in self.progressBar.track(descriptors):
            # todo try to run pca on features before codebooking
            code_vector = np.zeros(self.k_, dtype=float)
            if len(img_descriptors) != 0:
                quantized_descriptors = self.kmeans_.predict(img_descriptors)
                counter = collections.Counter(quantized_descriptors)
                for code in counter.keys():
                    code_vector[code] = counter[code]
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

        canvas.print_figure('../../results/example_codes.png')
