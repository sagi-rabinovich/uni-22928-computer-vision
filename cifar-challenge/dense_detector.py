import cv2


class DenseDetector:

    def __init__(self, radiuses):
        self._radiuses = radiuses
        self._cachedGrids = {}

    def detect(self, img, ignored):
        kp = self._cachedGrids.get(img.shape)
        if kp is not None:
            return kp

        kp = []
        for radius in self._radiuses:
            for y in range(radius, img.shape[0], radius):
                for x in range(radius, img.shape[1], radius):
                    kp.append(cv2.KeyPoint(x, y, radius * 2))

        self._cachedGrids[img.shape] = kp
        return kp
