import logging

import cv2
from sklearn import preprocessing

from dense_detector import DenseDetector


class FeatureExtractor:
    _useGrid = True
    _useSift = True

    def __init__(self, progressBar, min_patch_diameter=5):
        self._min_patch_diameter = min_patch_diameter
        self._detectors = [  # cv2.xfeatures2d.SIFT_create(),
            # cv2.xfeatures2d.SURF_create(),
            # cv2.MSER_create(),
            DenseDetector([3, 6, 8, 12, 16, 20], 0.3),
            # cv2.KAZE_create()
        ]
        self._descriptor = cv2.xfeatures2d.SURF_create()
        self._progressBar = progressBar
        self._logger = logging.getLogger('cifar-challenge.FeatureExtractor')


    def __describe(self, imageContexts):
        for imageContext in self._progressBar.track(imageContexts, suffix='Computing descriptors'):
            keyPoints, descriptors = self._descriptor.compute(imageContext.gray, imageContext.key_points)
            imageContext.features = preprocessing.normalize(descriptors, norm='l2')


    def extractAndCompute(self, imageContexts):
        for imageContext in self._progressBar.track(imageContexts, suffix='Detecting key points'):
            key_points = []
            for detector in self._detectors:
                detected_kp = detector.detect(imageContext.gray, None)
                if detected_kp is None:
                    self._logger.warn(
                        '[' + imageContext.image_path() + '] Did not detect any key points with detector of type: ' + detector.__class__.__name__)
                else:
                    key_points.extend(detected_kp)

            if len(key_points) == 0:
                raise Exception('Could not find key points for image: ' + imageContext.image_path())
            imageContext.key_points = key_points

        self.__describe(imageContexts)
