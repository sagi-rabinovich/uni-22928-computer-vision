from os import listdir
from os.path import isfile, join
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import ntpath
from code_book import CodeBook
from feature_descriptor import FeatureDescriptor
from image_context import ImageContext
from key_point_extractor import KeyPointExtractor

CIFAR_10_LABELS = ['airplane',
                   'automobile',
                   'bird',
                   'cat',
                   'deer',
                   'dog',
                   'frog',
                   'horse',
                   'ship',
                   'truck']


def array2Image(A, path):
    from PIL import Image
    im = Image.fromarray(A)
    im.save(path)


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def cifar10ToImages(dict, batch_name):
    data = dict.get('data')
    labels = dict['labels']
    data = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    labels = np.array(labels)

    for i in range(data.shape[0]):
        label = labels[i]
        labelWord = CIFAR_10_LABELS[label]
        directory = 'images/' + batch_name + '/' + str(label) + '.' + labelWord
        if not os.path.exists(directory):
            os.makedirs(directory)
        array2Image(data[i, :], directory + '/img_' + str(i) + '.png')


# dict = unpickle('./cifar-10-batches-py/test_batch')
# cifar10ToImages(dict, 'test_batch')

keyPointExtractor = KeyPointExtractor()
featureDescriptor = FeatureDescriptor()

imgDir = '.\\images\\test_batch\\2.bird'
imagePaths = [f for f in listdir(imgDir) if isfile(join(imgDir, f))]
imageContexts = []
i = 5
for imagePath in imagePaths:
    i -= 1
    if i <= 0:
        break

    img = cv2.imread(join(imgDir, imagePath))
    imageContext = ImageContext(img)
    imageContext.keyPoints = keyPointExtractor.extract(imageContext)
    imageContext.descriptors = featureDescriptor.describe(imageContext)
    imageContexts.append(imageContext)

codeBook = CodeBook(imageContexts)
codeBook.quantize(imageContexts)
codeBook.printExampleCodes(imageContexts, 10, 20)

# dummy = img.copy()
# cv2.drawKeypoints(imageContext.gray, imageContext.keyPoints, dummy, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
# cv2.imwrite('sift_keypoints.jpg', dummy)  # features = FeatureDescriptor().describe(imageContext, keyPoints)
print('done')

# dummy = image.copy()
# cv2.drawKeypoints(gray, kp, dummy, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite('sift_keypoints.jpg', dummy)
