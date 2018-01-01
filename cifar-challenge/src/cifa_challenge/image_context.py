import cv2


class ImageContext:
    def __init__(self, source_file, index, image, label):
        self.index = index
        self.source_file = source_file
        self.original = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.label = label
        self.key_points = None
        self.features = None
        self.quantized_descriptors = None
        self.image_descriptor = None
        self.code_vector = None

    def image_path(self):
        return self.source_file + ':' + str(self.index)
