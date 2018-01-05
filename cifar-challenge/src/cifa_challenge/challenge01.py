from feature_extractor import FeatureExtractor

from classifier import Classifier
from image_dataset import ImageDataset
from pipeline.code_book import CodeBook
from progress_bar import ProgressBar


def pickle_results(X, Y, file_name):
    for xx in X:
        xx.prepare_to_pickle()
    for yy in Y:
        yy.prepare_to_pickle()
    image_dataset.pickle_obj((X, Y), file_name)
    for xx in X:
        xx.unpickle()
    for yy in Y:
        yy.unpickle()


def unpickle_results(file_name):
    X, Y = image_dataset.unpickle_obj(file_name)
    for xx in X:
        xx.unpickle()
    for yy in Y:
        yy.unpickle()
    return X, Y

progress_bar = ProgressBar()
feature_extractor = FeatureExtractor(progress_bar)
image_dataset = ImageDataset()

#
# imgDir = '.\\images\\test_batch\\2.bird'
# imagePaths = [f for f in listdir(imgDir) if isfile(join(imgDir, f))]
# image_contexts = []
# i = 5
# for imagePath in imagePaths:
#     i -= 1
#     if i <= 0:
#         break
#     img = cv2.imread(join(imgDir, imagePath))
#     imageContext = ImageContext(img)
#     imageContext.key_points = keyPointExtractor.extract(imageContext)
#     imageContext.features = featureDescriptor.describe(imageContext)
#     image_contexts.append(imageContext)
DATA_BATCH_1 = 'data_batch_1'

image_contexts = image_dataset.load_training_data(batch=DATA_BATCH_1)[0:1000]
test_image_contexts = image_dataset.load_test_data()
progress_bar.prefix = 'Extracting features for training images'
feature_extractor.extractAndCompute(image_contexts)
progress_bar.prefix = 'Extracting features for test images'
feature_extractor.extractAndCompute(test_image_contexts)
# pickle_results(image_contexts, test_image_contexts, 'all_data')
#
# image_contexts = image_dataset.load_training_data(batch=DATA_BATCH_1)
# test_image_contexts = image_dataset.load_test_data()
# progress_bar.prefix = 'Extracting features for training images'
# feature_extractor.extractAndCompute(image_contexts)
# progress_bar.prefix = 'Extracting features for test images'
# feature_extractor.extractAndCompute(test_image_contexts)
# pickle_results(image_contexts, test_image_contexts, 'results_from_batch_1')

# image_contexts, test_image_contexts = unpickle_results('all_data')
# image_contexts, test_image_contexts = unpickle_results('results_from_batch_1')

progress_bar.prefix = 'Building code book'
code_book = CodeBook(progress_bar).fit(image_contexts)
progress_bar.prefix = 'Computing code vectors for test images'
code_book.compute_for_test_images(test_image_contexts)
progress_bar.prefix = 'Training classifier'
classifier = Classifier().learn(image_contexts)
progress_bar.prefix = 'Testing on test data'
score = classifier.score(test_image_contexts)
classifier.plot_confusion_matrix(test_image_contexts, classifier.predict(test_image_contexts),
                                 image_dataset.CIFAR_10_LABELS)

print("The score: " + str(score))

#
# def print_predictions():
#     logger.info('Printing predictions')
#     plt.ioff()
#
#     figure = Figure(figsize=(10, 10))
#     canvas = FigureCanvas(figure)
#     nearest_neighbors_count = 10
#     nearest_neighbors_columns = 5
#     nearest_neighbors_rows = int(np.math.ceil(float(nearest_neighbors_count) / nearest_neighbors_columns))
#
#     predictions = classifier.predict(test_image_contexts)
#     test_imgs_count = len(test_image_contexts)
#
#     for test_i in range(test_imgs_count):
#         testImageContext = test_image_contexts[test_i]
#         predictedLabelNumber = predictions[test_i]
#         predictedLabelText = image_dataset.label(predictedLabelNumber)
#         test_im_ax = figure.add_subplot(1, test_imgs_count, test_i + 1)
#         # ax = figure.add_subplot(1 + nearest_neighbors_rows, test_imgs_count*nearest_neighbors_columns, test_i*nearest_neighbors_columns)
#         test_im_ax.set_axis_off()
#         test_im_ax.imshow(testImageContext.original, interpolation='nearest')
#         table = test_im_ax.table(cellText=[[predictedLabelText, image_dataset.label(testImageContext.label)]],
#                                  colLabels=['Predicted', 'Correct'],
#                                  loc='bottom')
#
#         table.scale(1, 4)
#         table.set_fontsize(14)
#
#     plt.tight_layout()
#     canvas.print_figure('../../results/prediction.png', bbox_inches='tight', dpi=100)
#
#     for test_i in range(test_imgs_count):
#         testImageContext = test_image_contexts[test_i]
#         neighbors = classifier.knn(testImageContext, nearest_neighbors_count)
#
#         figure = Figure(figsize=(10, 10))
#         canvas = FigureCanvas(figure)
#         for ni in range(len(neighbors)):
#             neighbor = neighbors[ni]
#             neighbor_img_ax = figure.add_subplot(nearest_neighbors_rows, nearest_neighbors_columns, ni + 1)
#             neighbor_img_ax.set_axis_off()
#             neighbor_img_ax.spines['bottom'].set_color('0.5')
#             neighbor_img_ax.spines['top'].set_color('0.5')
#             neighbor_img_ax.spines['right'].set_color('0.5')
#             neighbor_img_ax.spines['left'].set_color('0.5')
#             neighbor_img_ax.imshow(neighbor.original, interpolation='nearest')
#         plt.tight_layout()
#         canvas.print_figure('../../results/prediction_' + str(test_i) + '_nn.png', bbox_inches='tight', dpi=100)


# code_book.printExampleCodes(image_contexts, 10, 20)

# dummy = img.copy()
# cv2.drawKeypoints(imageContext.gray, imageContext.key_points, dummy, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
# cv2.imwrite('sift_keypoints.jpg', dummy)  # features = FeatureDescriptor().describe(imageContext, key_points)

logger.info('Done challenge')
# dummy = image.copy()
# cv2.drawKeypoints(gray, kp, dummy, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite('sift_keypoints.jpg', dummy)
