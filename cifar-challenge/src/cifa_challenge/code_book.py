import collections
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
        print('Building code book')
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
        print('computing kmeans for features' + str(features.shape))
        self._kmeans = MiniBatchKMeans(n_clusters=self._k, verbose=1)

        def __build_(image_ctx):
            self._kmeans.fit(image_ctx.features)

        self._progressBar.forEach(image_contexts, __build)

    def computeCodeVector(self, imageContexts):
        def __computeCodeVector(imageContext):
            # todo try to run pca on features before codebooking
            # todo whitening features multiple times
            # todo optimize - memory is not freed in image context and all sub calculations are kept
            # imageContext.quantized_descriptors = vq(whiten(imageContext.features), self._codebook, False)
            imageContext.quantizedDescriptors = self._kmeans.predict(imageContext.features)
            counter = collections.Counter(imageContext.quantizedDescriptors)
            imageContext.codeVector = np.zeros(self._k, dtype=float)
            for code in counter.keys():
                imageContext.codeVector[code] = counter[code]

        self._progressBar.forEach(imageContexts, __computeCodeVector)

    def printExampleCodes(self, imageContexts, sampleCodesCount, samplesImagesPerCode):
        codes = np.random.choice(range(self._k), sampleCodesCount)

        plt.ioff()
        figure = Figure(figsize=(10, 10))
        canvas = FigureCanvas(figure)

        originals = [imageContext.original for imageContext in imageContexts]
        repeats = [len(imageContext.features) for imageContext in imageContexts]
        images = np.repeat(originals, repeats, axis=0)
        keyPoints = np.concatenate([imageContext.keyPoints for imageContext in imageContexts])
        quantizedDescriptors = np.concatenate([imageContext.quantizedDescriptors for imageContext in imageContexts])

        # codes
        for row in range(sampleCodesCount):
            code = codes[row]
            b = quantizedDescriptors == code
            clusteredFeatures = keyPoints[b]
            samples = np.random.choice(range(len(clusteredFeatures)), samplesImagesPerCode)
            sampleKeyPoints = clusteredFeatures[samples]
            sampleImages = images[b][samples]
            for column in range(min(len(samples), samplesImagesPerCode)):
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
