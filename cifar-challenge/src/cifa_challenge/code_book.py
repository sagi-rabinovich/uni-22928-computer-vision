import collections
import logging
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from numpy import random
# from scipy.cluster.vq import kmeans
# from scipy.cluster.vq import whiten
# from scipy.cluster.vq import vq
from sklearn.cluster import MiniBatchKMeans


class CodeBook:

    def __init__(self, progressBar, imageContexts):
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

        self.__build(imageContexts)

    def __build(self, image_contexts):
        self._logger.info('Building code book')
        random.seed((1000, 2000))
        if len(image_contexts) <= 0:
            raise Exception('Empty argument image_contexts')


        labelCount = 10
        # self._k = int(math.ceil(labelCount * np.average([len(featuresForImage) for featuresForImage in features])))
        self._k = 100
        # todo handle cases where there are photos without features?

        #
        # whitened = whiten(features)
        # codebook, distortion = kmeans(whitened, k)
        # self._codebook = codebook
        self._progressBar.prefix = 'computing kmeans'
        self._kmeans = MiniBatchKMeans(n_clusters=self._k)

        split = len(image_contexts) / 10000 + 1
        for batch in self._progressBar.track(np.array_split(image_contexts, split)):
            features = np.concatenate([img_ctx.features for img_ctx in batch])
            self._kmeans.fit(features)

    def computeCodeVector(self, imageContexts):
        for imageContext in self._progressBar.track(imageContexts):
            # todo try to run pca on features before codebooking
            # todo whitening features multiple times
            # todo optimize - memory is not freed in image context and all sub calculations are kept
            # imageContext.quantized_descriptors = vq(whiten(imageContext.features), self._codebook, False)
            imageContext.quantized_descriptors = self._kmeans.predict(imageContext.features)
            counter = collections.Counter(imageContext.quantized_descriptors)
            imageContext.code_vector = np.zeros(self._k, dtype=float)
            for code in counter.keys():
                imageContext.code_vector[code] = counter[code]

    def printExampleCodes(self, imageContexts, sampleCodesCount, samplesImagesPerCode):
        self._logger.info('Printing example codes')
        codes = np.random.choice(range(self._k), sampleCodesCount)

        plt.ioff()
        figure = Figure(figsize=(10, 10))
        canvas = FigureCanvas(figure)

        originals = [imageContext.original for imageContext in imageContexts]
        repeats = [len(imageContext.features) for imageContext in imageContexts]
        images_repeated = np.repeat(originals, repeats, axis=0)
        all_key_points = np.concatenate([imageContext.key_points for imageContext in imageContexts])
        all_quantized_descriptors = np.concatenate(
            [imageContext.quantized_descriptors for imageContext in imageContexts])

        # codes
        for row in range(sampleCodesCount):
            code = codes[row]
            b = all_quantized_descriptors == code
            clusteredFeatures = all_key_points[b]
            samples = np.random.choice(range(len(clusteredFeatures)), samplesImagesPerCode)
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

                figIndex = row * samplesImagesPerCode + column + 1
                ax = figure.add_subplot(sampleCodesCount, samplesImagesPerCode, figIndex)
                ax.set_axis_off()
                ax.imshow(kpImage, interpolation='nearest')

        canvas.print_figure('../../results/example_codes.png')
