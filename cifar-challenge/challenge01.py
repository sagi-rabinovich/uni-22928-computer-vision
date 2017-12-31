import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_template import FigureCanvas
from matplotlib.figure import Figure

from classifier import Classifier
from code_book import CodeBook
from feature_descriptor import FeatureDescriptor
from image_dataset import ImageDataset
from key_point_extractor import KeyPointExtractor
from progress_bar import ProgressBar

progressBar = ProgressBar()
keyPointExtractor = KeyPointExtractor(progressBar)
featureDescriptor = FeatureDescriptor(progressBar)
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
#     imageContext.keyPoints = keyPointExtractor.extract(imageContext)
#     imageContext.descriptors = featureDescriptor.describe(imageContext)
#     imageContexts.append(imageContext)

imageContexts = imageDataset.load_training_data(10)
testImageContexts = np.random.choice(imageDataset.load_test_data(), 10)
progressBar.prefix = 'Extracting features from training images'
keyPointExtractor.extract(imageContexts)
progressBar.prefix = 'Computing descriptors for training images'
featureDescriptor.describe(imageContexts)
progressBar.prefix = 'Extracting features from test images'
keyPointExtractor.extract(testImageContexts)
progressBar.prefix = 'Computing descriptors for test images'
featureDescriptor.describe(testImageContexts)

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
plt.ioff()
figure = Figure(figsize=(10, 10))
canvas = FigureCanvas(figure)

i = 0

predictions = classifier.predict(testImageContexts)
for testImageContext in testImageContexts:
    i += 1
    predictedLabelNumber = predictions[i - 1]
    predictedLabelText = imageDataset.label(predictedLabelNumber)
    ax = figure.add_subplot(1, len(testImageContexts), i)
    ax.set_axis_off()
    ax.imshow(testImageContext.original, interpolation='nearest')
    table = ax.table(cellText=[[predictedLabelText, imageDataset.label(testImageContext.label)]],
                     colLabels=['Predicted', 'Correct'],
                     loc='bottom')

    table.scale(1, 4)

canvas.print_figure('prediction.png')

print("The score: " + str(score))
# codeBook.printExampleCodes(imageContexts, 10, 20)

# dummy = img.copy()
# cv2.drawKeypoints(imageContext.gray, imageContext.keyPoints, dummy, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
# cv2.imwrite('sift_keypoints.jpg', dummy)  # features = FeatureDescriptor().describe(imageContext, keyPoints)
print('done')

# dummy = image.copy()
# cv2.drawKeypoints(gray, kp, dummy, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite('sift_keypoints.jpg', dummy)
