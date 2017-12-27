from os import listdir
from os.path import isfile, join

import cv2

from classifier import Classifier
from code_book import CodeBook
from feature_descriptor import FeatureDescriptor
from image_context import ImageContext
from key_point_extractor import KeyPointExtractor

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

testImageContexts = []

codeBook = CodeBook(imageContexts)
codeBook.computeCodeVector(imageContexts)
Classifier().learn(imageContexts).score()
# codeBook.printExampleCodes(imageContexts, 10, 20)

# dummy = img.copy()
# cv2.drawKeypoints(imageContext.gray, imageContext.keyPoints, dummy, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
# cv2.imwrite('sift_keypoints.jpg', dummy)  # features = FeatureDescriptor().describe(imageContext, keyPoints)
print('done')

# dummy = image.copy()
# cv2.drawKeypoints(gray, kp, dummy, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite('sift_keypoints.jpg', dummy)

