import cv2
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

from cifa_challenge.progress_bar import ProgressBar
from color_descriptor import ColorSiftDescriptor, ColorSurfDescriptor


class FeatureDescriptor(BaseEstimator, TransformerMixin):
    def __init__(self, progressBar=ProgressBar(), descriptor='surf'):
        self.progressBar = progressBar
        self.descriptor = descriptor

    def fit(self, images, y=None):
        return self

    def get_descriptor_(self):
        if self.descriptor == 'surf':
            return cv2.xfeatures2d.SURF_create()
        elif self.descriptor == 'sift':
            return cv2.xfeatures2d.SIFT_create()
        elif self.descriptor == 'kaze':
            return cv2.KAZE_create()
        elif self.descriptor == 'color-sift':
            return ColorSiftDescriptor()
        elif self.descriptor == 'color-surf':
            return ColorSurfDescriptor()
        else:
            raise RuntimeError('Unknown descriptor: ' + str(self.descriptor))

    def transform(self, X):
        images = X[0]
        kps = X[1]
        descriptors = []

        i = 0
        descriptor = self.get_descriptor_()
        for img in self.progressBar.track(images):
            img_key_points, img_descriptors = descriptor.compute(img, kps[i])
            if img_descriptors is None:
                img_descriptors = np.empty(0)
            descriptors.append(img_descriptors)
            i += 1
        return images, descriptors
