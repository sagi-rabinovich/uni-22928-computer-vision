import cv2
import numpy as np


class KeyPointExtractor:
    _useGrid = True
    _useSift = True

    _gridKeyPointSizeRange = range(5, 6, 1)
    _gridCellInterval = 4

    def __init__(self, progressBar):
        self._sift = cv2.xfeatures2d.SIFT_create()
        self._progressBar = progressBar

    def extract(self, imageContexts):

        def _extract(imageContext):

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
            imageContext.keyPoints = kp

        self._progressBar.forEach(imageContexts, _extract)
