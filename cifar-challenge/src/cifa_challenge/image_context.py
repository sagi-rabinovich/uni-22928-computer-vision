import cv2


class ImageContext:
    def __init__(self, image, label):
        self.keyPoints = None
        self.imageDescriptor = None
        self.descriptors = []
        self.quantizedDescriptors = None
        self.original = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.label = label
        self.codeVector = None
