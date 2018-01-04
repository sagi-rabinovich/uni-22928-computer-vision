import cv2
import numpy as np


class PerColorChannelDescriptor(object):
    def __init__(self):
        self.descriptor_ = None

    def compute(self, image, kp):
        if len(image.shape) != 3 or image.shape[2] < 2:
            raise RuntimeError(
                'image should be a 3D Array of shape (height, width, nchannels] where nchannels >= 2')
        descriptors_by_channel = []
        for i in range(image.shape[2]):
            ignore, desc = self.descriptor_.compute(image[:, :, i], kp)
            descriptors_by_channel.append(desc)
        return kp, np.concatenate(descriptors_by_channel, axis=1)


class ColorSiftDescriptor(PerColorChannelDescriptor):
    def __init__(self):
        self.descriptor_ = cv2.xfeatures2d.SIFT_create()


class ColorSurfDescriptor(PerColorChannelDescriptor):
    def __init__(self):
        self.descriptor_ = cv2.xfeatures2d.SURF_create()
