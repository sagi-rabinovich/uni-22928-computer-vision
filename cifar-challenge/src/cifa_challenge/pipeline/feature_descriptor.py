import cv2
from sklearn.base import TransformerMixin, BaseEstimator

from cifa_challenge.progress_bar import ProgressBar


class FeatureDescriptor(BaseEstimator, TransformerMixin):
    def __init__(self, progressBar=ProgressBar(), descriptor=cv2.xfeatures2d.SURF_create()):
        self.progressBar = progressBar
        self.descriptor = descriptor

    def fit(self, images, y=None):
        return self

    def transform(self, X):
        images = X[0]
        kps = X[1]
        descriptors = []

        i = 0
        for img in self.progressBar.track(images):
            img_key_points, img_descriptors = self.descriptor.compute(img, kps[i])
            descriptors.append(img_descriptors)
            i += 1
        return descriptors
