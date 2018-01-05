import cv2
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class DenseDetector(BaseEstimator, TransformerMixin):
    """ An example transformer that returns the element-wise square root..
    Parameters
    ----------
    radiuses : list of integers, optional
        radiuses of the patches to extract
    overlap : float, optional
        Approximate overlap between the patches, in percents.

    Attributes
    ----------
    input_shape : tuple
        The shape the data passed to :meth:`fit`
    """

    def __init__(self, radiuses=[3], overlap=0.3):
        self.radiuses = radiuses
        self.overlap = overlap
        self._cachedGrids = {}
        self.__fitted_images = None
        self.__detected_kps = None

    def fit(self, images, y=None):
        return self

    def transform(self, images):
        kps = []
        for im in images:
            kps.append(self.__detect(im))
        return images, kps

    def __detect(self, img):
        img_shape = img.shape
        kp = self._cachedGrids.get(img_shape)
        if kp is not None:
            return kp

        kp = []
        for radius in self.radiuses:
            step = int(np.math.ceil(radius * 2) * (1 - self.overlap))
            for y in range(radius, img_shape[0], step):
                for x in range(radius, img_shape[1], step):
                    kp.append(cv2.KeyPoint(x, y, radius * 2))

        self._cachedGrids[img_shape] = kp
        return kp
