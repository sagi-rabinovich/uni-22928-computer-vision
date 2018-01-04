import cv2
from sklearn.base import TransformerMixin, BaseEstimator

from cifa_challenge.progress_bar import ProgressBar


class FeatureDetector(BaseEstimator, TransformerMixin):
    def __init__(self, progressBar=ProgressBar(), detector='sift'):
        self.descriptor_ = None
        self.progressBar = progressBar
        self.detector = detector
        self.detector_ = None

    def fit(self, images, y=None):
        if self.detector == 'surf':
            self.detector_ = cv2.xfeatures2d.SURF_create()
        elif self.detector == 'sift':
            self.detector_ = cv2.xfeatures2d.SIFT_create()
        elif self.detector == 'kaze':
            self.detector_ = cv2.KAZE_create()
        else:
            raise RuntimeError('Unknown detector: ' + str(self.detector))
        return self

    def transform(self, images):
        kps = []
        for img in self.progressBar.track(images):
            img_key_points = self.detector_.detect(img.gray)
            kps.append(img_key_points)
        return images, kps
