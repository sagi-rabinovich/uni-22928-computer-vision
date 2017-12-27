import math
import numpy as np
from matplotlib.figure import Figure
from numpy import random
from matplotlib import pyplot as plt
# from scipy.cluster.vq import kmeans
# from scipy.cluster.vq import whiten
# from scipy.cluster.vq import vq
from sklearn.cluster import KMeans
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class CodeBook:
    def __init__(self, imageContexts):
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
        self.__build(imageContexts)

    def __build(self, imageContexts):
        random.seed((1000, 2000))
        if len(imageContexts) <= 0:
            raise 'Empty argument imageContexts'

        self._k = len(imageContexts[0].descriptors) / 2
        features = [imageContext.descriptors for imageContext in imageContexts]
        features = np.concatenate(features)
        #
        # whitened = whiten(features)
        # codebook, distortion = kmeans(whitened, k)
        # self._codebook = codebook
        self._kmeans = KMeans(n_clusters=self._k).fit(features)

    def quantize(self, imageContexts):
        for imageContext in imageContexts:
            # todo try to run pca on descriptors before codebooking
            # todo whitening descriptors multiple times
            # todo optimize - memory is not freed in image context and all sub calculations are kept
            # imageContext.quantizedDescriptors = vq(whiten(imageContext.descriptors), self._codebook, False)
            imageContext.quantizedDescriptors = self._kmeans.predict(imageContext.descriptors)


    def printExampleCodes(self, imageContexts, labelsCount, samplePerLabel):
        labels = np.random.choice(range(self._k), labelsCount)

        plt.ioff()
        figure = Figure()
        canvas = FigureCanvas(figure)

        originals = [imageContext.original for imageContext in imageContexts]
        repeats = [len(imageContext.descriptors) for imageContext in imageContexts]
        images = np.repeat(originals, repeats, axis=0)
        keyPoints = np.concatenate([imageContext.keyPoints for imageContext in imageContexts])
        quantizedDescriptors = np.concatenate([imageContext.quantizedDescriptors for imageContext in imageContexts])

        # codes
        for row in range(labelsCount):
            label = labels[row]
            b = quantizedDescriptors == label
            clusteredKeyPoints = keyPoints[b]
            samples = np.random.choice(range(len(clusteredKeyPoints)), samplePerLabel)
            sampleKeyPoints = clusteredKeyPoints[samples]
            sampleImages = images[b][samples]
            for column in range(min(len(samples), samplePerLabel)):
                sampleImage = sampleImages[column]
                sampleKp = sampleKeyPoints[column]
                radius = int(math.ceil(sampleKp.size / 2.0))
                x = int(sampleKp.pt[0])
                y = int(sampleKp.pt[1])
                kpImage = sampleImage[max(x - radius, 0):min(x + radius, sampleImage.shape[0]),
                          max(y - radius, 0):min(y + radius, sampleImage.shape[1])]
                figIndex = row * samplePerLabel + column + 1
                ax = figure.add_subplot(labelsCount, samplePerLabel, figIndex)
                ax.set_axis_off()
                ax.imshow(kpImage)

        figure.tight_layout()
        canvas.print_figure('foo.png')
