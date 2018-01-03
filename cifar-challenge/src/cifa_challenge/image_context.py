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

    def prepare_to_pickle(self):
        self.key_points = [(point.pt, point.size, point.angle, point.response, point.octave,
                            point.class_id) for point in self.key_points]

    def unpickle(self):
        self.key_points = [cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2],
                                        _response=point[3], _octave=point[4], _class_id=point[5])
                           for point in self.key_points]
