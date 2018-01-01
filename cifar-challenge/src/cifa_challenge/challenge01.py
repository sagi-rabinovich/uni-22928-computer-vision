import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_template import FigureCanvas
from matplotlib.figure import Figure

from classifier import Classifier
from code_book import CodeBook
from feature_extractor import FeatureExtractor
from image_dataset import ImageDataset
from progress_bar import ProgressBar

# Logging
# create logger with 'spam_application'
logger = logging.getLogger('cifar-challenge')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('spam.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

progressBar = ProgressBar()
featureExtractor = FeatureExtractor(progressBar)
imageDataset = ImageDataset()
#
# imgDir = '.\\images\\test_batch\\2.bird'
# imagePaths = [f for f in listdir(imgDir) if isfile(join(imgDir, f))]
# imageContexts = []
# i = 5
# for imagePath in imagePaths:
#     i -= 1
#     if i <= 0:
#         break
#     img = cv2.imread(join(imgDir, imagePath))
#     imageContext = ImageContext(img)
#     imageContext.key_points = keyPointExtractor.extract(imageContext)
#     imageContext.features = featureDescriptor.describe(imageContext)
#     imageContexts.append(imageContext)

imageContexts = imageDataset.load_training_data(1000)
testImageContexts = np.random.choice(imageDataset.load_test_data(), 10)
progressBar.prefix = 'Extracting features from training images'
featureExtractor.extractAndCompute(imageContexts)
progressBar.prefix = 'Extracting features from test images'
featureExtractor.extractAndCompute(testImageContexts)

progressBar.prefix = 'Building code book'
codeBook = CodeBook(progressBar, imageContexts)
progressBar.prefix = 'Computing code vectors for training images'
codeBook.computeCodeVector(imageContexts)
progressBar.prefix = 'Computing code vectors for test images'
codeBook.computeCodeVector(testImageContexts)
progressBar.prefix = 'Training classifier'
classifier = Classifier().learn(imageContexts)
progressBar.prefix = 'Testing on test data'
score = classifier.score(testImageContexts)

## Print predictions
logger.info('Printing predictions')
plt.ioff()

figure = Figure(figsize=(10, 10))
canvas = FigureCanvas(figure)
nearest_neighbors_count = 10
nearest_neighbors_columns = 5
nearest_neighbors_rows = int(np.math.ceil(float(nearest_neighbors_count) / nearest_neighbors_columns))

predictions = classifier.predict(testImageContexts)
test_imgs_count = len(testImageContexts)

for test_i in range(test_imgs_count):
    testImageContext = testImageContexts[test_i]
    predictedLabelNumber = predictions[test_i]
    predictedLabelText = imageDataset.label(predictedLabelNumber)
    test_im_ax = figure.add_subplot(1, test_imgs_count, test_i + 1)
    # ax = figure.add_subplot(1 + nearest_neighbors_rows, test_imgs_count*nearest_neighbors_columns, test_i*nearest_neighbors_columns)
    test_im_ax.set_axis_off()
    test_im_ax.imshow(testImageContext.original, interpolation='nearest')
    table = test_im_ax.table(cellText=[[predictedLabelText, imageDataset.label(testImageContext.label)]],
                             colLabels=['Predicted', 'Correct'],
                             loc='bottom')

    table.scale(1, 4)
    table.set_fontsize(14)

plt.tight_layout()
canvas.print_figure('results/prediction.png', bbox_inches='tight', dpi=100)

for test_i in range(test_imgs_count):
    testImageContext = testImageContexts[test_i]
    neighbors = classifier.knn(testImageContext, nearest_neighbors_count)

    figure = Figure(figsize=(10, 10))
    canvas = FigureCanvas(figure)
    for ni in range(len(neighbors)):
        neighbor = neighbors[ni]
        neighbor_img_ax = figure.add_subplot(nearest_neighbors_rows, nearest_neighbors_columns, ni + 1)
        neighbor_img_ax.set_axis_off()
        neighbor_img_ax.spines['bottom'].set_color('0.5')
        neighbor_img_ax.spines['top'].set_color('0.5')
        neighbor_img_ax.spines['right'].set_color('0.5')
        neighbor_img_ax.spines['left'].set_color('0.5')
        neighbor_img_ax.imshow(neighbor.original, interpolation='nearest')
    plt.tight_layout()
    canvas.print_figure('results/prediction_' + str(test_i) + '_nn.png', bbox_inches='tight', dpi=100)


print("The score: " + str(score))
# codeBook.printExampleCodes(imageContexts, 10, 20)

# dummy = img.copy()
# cv2.drawKeypoints(imageContext.gray, imageContext.key_points, dummy, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
# cv2.imwrite('sift_keypoints.jpg', dummy)  # features = FeatureDescriptor().describe(imageContext, key_points)
print('done')

# dummy = image.copy()
# cv2.drawKeypoints(gray, kp, dummy, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite('sift_keypoints.jpg', dummy)
