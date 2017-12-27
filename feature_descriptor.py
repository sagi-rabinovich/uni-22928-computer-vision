import cv2


class FeatureDescriptor:

    def __init__(self):
        self._sift = cv2.xfeatures2d.SIFT_create()

    def describe(self, imageContext):
        keyPoints, descriptors = self._sift.compute(imageContext.gray, imageContext.keyPoints)
        return descriptors
