import cv2
import numpy as np

class DenseDetector:

    def __init__(self, radiuses, overlap):
        self._radiuses = radiuses
        self._overlap = overlap
        self._cachedGrids = {}

    def detect(self, img, ignored):
        kp = self._cachedGrids.get(img.shape)
        if kp is not None:
            return kp

        kp = []
        for radius in self._radiuses:
            step = int(np.math.ceil(radius * 2) * (1 - self._overlap))
            for y in range(radius, img.shape[0], step):
                for x in range(radius, img.shape[1], step):
                    kp.append(cv2.KeyPoint(x, y, radius * 2))

        self._cachedGrids[img.shape] = kp
        return kp
