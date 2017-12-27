import cv2
import numpy as np


class ImageContext:
    def __init__(self, image):
        self.keyPoints = None
        self.imageDescriptor = None
        self.descriptors = []
        self.quantizedDescriptors = None
        self.original = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
