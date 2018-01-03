import collections
import logging
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from memory_profiler import profile
from numpy import random
# from scipy.cluster.vq import kmeans
# from scipy.cluster.vq import whiten
# from scipy.cluster.vq import vq
from sklearn.cluster import MiniBatchKMeans

from my_utils import flatmap


class CodeBook:

    def __init__(self, progressBar):
        self._logger = logging.getLogger('cifar-challenge.CodeBook')
        """
        _codebook : ndarray
           A k by N array of k centroids. The i'th centroid
           codebook[i] is represented with the code i. The centroids
           and codes generated represent the lowest distortion seen,
           not necessarily the globally minimal distortion.
        """
        self._codebook = None
        self._kmeans = None
        self._k = -1
        self._progressBar = progressBar
        self._labels_ = []

    @profile
    def __build(self, image_contexts):
        random.seed((1000, 2000))
        if len(image_contexts) <= 0:
            raise Exception('Empty argument image_contexts')

        labelCount = 10
        self._k = int(0.7 *
                      math.ceil(
                          labelCount * np.mean([len(image_context.features) for image_context in image_contexts])))
        self._logger.info('Building code book [k=' + str(self._k) + ']')

        # self._k = 100
        # todo handle cases where there are photos without features?

        #
        # whitened = whiten(features)
        # codebook, distortion = kmeans(whitened, k)
        # self._codebook = codebook
        self._progressBar.prefix = 'computing kmeans'
        self._kmeans = MiniBatchKMeans(n_clusters=self._k, verbose=0)

        split = len(image_contexts) / 5000 + 1
        for batch in self._progressBar.track(np.array_split(image_contexts, split)):
            features = list(flatmap(lambda x: x.features, batch))
            self._kmeans.partial_fit(features)
            self._labels_.extend(self._kmeans.labels_)

    @profile
    def fit(self, image_contexts):
        self.__build(image_contexts)

        feature_start_index = 0
        for img_ctx in self._progressBar.track(image_contexts):
            # todo try to run pca on features before codebooking
            # todo whitening features multiple times
            # todo optimize - memory is not freed in image context and all sub calculations are kept
            # imageContext.quantized_descriptors = vq(whiten(imageContext.features), self._codebook, False)
            current_feature_count = len(img_ctx.features)
            img_ctx.quantized_descriptors = self._labels_[
                                            feature_start_index:feature_start_index + current_feature_count]
            feature_start_index += current_feature_count
        self.__compute_code_vector(image_contexts)

        return self

    def __compute_code_vector(self, image_contexts):
        self._progressBar.prefix += ' - computing code vectors'
        for imageContext in self._progressBar.track(image_contexts):
            counter = collections.Counter(imageContext.quantized_descriptors)
            imageContext.code_vector = np.zeros(self._k, dtype=float)
            for code in counter.keys():
                imageContext.code_vector[code] = counter[code]

    @profile
    def compute_for_test_images(self, test_image_context):
        for test_img_ctx in self._progressBar.track(test_image_context):
            test_img_ctx.quantized_descriptors = self._kmeans.predict(test_img_ctx.features)
        self.__compute_code_vector(test_image_context)

    def printExampleCodes(self, image_contexts, sample_codes_count, samples_images_per_code):
        self._logger.info('Printing example codes')
        codes = np.random.choice(range(self._k), sample_codes_count)

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
