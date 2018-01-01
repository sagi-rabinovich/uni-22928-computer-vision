import cv2

from progress_bar import ProgressBar


class FeatureDescriptor:

    def __init__(self, progressBar=ProgressBar()):
        self._sift = cv2.xfeatures2d.SIFT_create()
        self._progressBar = progressBar

    def describe(self, imageContexts):
        def _describe(imageContext):
            keyPoints, descriptors = self._sift.compute(imageContext.gray, imageContext.keyPoints)
            imageContext.descriptors = descriptors

        self._progressBar.forEach(imageContexts, _describe)
