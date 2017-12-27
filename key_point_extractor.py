import cv2
import numpy as np


class KeyPointExtractor:
    _useGrid = True
    _useSift = True

    _gridKeyPointSizeRange = range(5, 5, 0)
    _gridCellInterval = 2

    def __init__(self):
        self._sift = cv2.xfeatures2d.SIFT_create()

    def extract(self, imageContext):
        kp = self._sift.detect(imageContext.gray, None)

        def extractGridKeyPoints(width, height, cellSize):
            grid = []
            for y in range(0, width, self._gridCellInterval):
                for x in range(0, height, self._gridCellInterval):
                    grid.append(cv2.KeyPoint(x, y, cellSize))
            return grid

        shape = imageContext.gray.shape
        count = shape[0] * shape[1] / (self._gridCellInterval * self._gridCellInterval)
        for gridSize in self._gridKeyPointSizeRange:
            gridKp = extractGridKeyPoints(shape[1], shape[0], gridSize)
            kp = np.concatenate((kp, gridKp))
        return kp
