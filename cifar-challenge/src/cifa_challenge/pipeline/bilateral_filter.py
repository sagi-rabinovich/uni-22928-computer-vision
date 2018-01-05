import cv2
from sklearn.base import TransformerMixin, BaseEstimator


class BilateralFilter(BaseEstimator, TransformerMixin):

    def fit(self, images, y=None):
        return self

    def transform(self, images):
        return [cv2.bilateralFilter(X, 4, 50, 50) for X in images]
