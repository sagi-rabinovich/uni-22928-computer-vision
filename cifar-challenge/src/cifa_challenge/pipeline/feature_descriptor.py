import cv2
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

from cifa_challenge.progress_bar import ProgressBar
from color_descriptor import ColorSiftDescriptor, ColorSurfDescriptor


class FeatureDescriptor(BaseEstimator, TransformerMixin):
    def __init__(self, progressBar=ProgressBar(), descriptor='surf'):
        self.descriptor_ = None
        self.progressBar = progressBar
        self.descriptor = descriptor

    def fit(self, images, y=None):
        if self.descriptor == 'surf':
            self.descriptor_ = cv2.xfeatures2d.SURF_create()
        elif self.descriptor == 'sift':
            self.descriptor_ = cv2.xfeatures2d.SIFT_create()
        elif self.descriptor == 'kaze':
            self.descriptor_ = cv2.KAZE_create()
        elif self.descriptor == 'color-sift':
            self.descriptor_ = ColorSiftDescriptor()
        elif self.descriptor == 'color-surf':
            self.descriptor_ = ColorSurfDescriptor()
        else:
            raise RuntimeError('Unknown descriptor: ' + str(self.descriptor))
        return self

    def transform(self, X):
        images = X[0]
        kps = X[1]
        descriptors = []

        i = 0
        for img in self.progressBar.track(images):
            img_key_points, img_descriptors = self.descriptor_.compute(img, kps[i])
            if img_descriptors is None:
                img_descriptors = np.empty(0)
            descriptors.append(img_descriptors)
            i += 1
        return images, descriptors
