import itertools

import numpy as np
import sklearn
from matplotlib import pyplot as plt
from matplotlib.backends.backend_template import FigureCanvas
from matplotlib.figure import Figure
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

from cifa_challenge.image_grid_plot import plot_image_grid
from cifa_challenge.pipeline.code_book import CodeBook
from cifa_challenge.pipeline.color_space_transformer import ColorSpaceTransformer
from cifa_challenge.pipeline.dense_detector import DenseDetector
from cifa_challenge.pipeline.feature_descriptor import FeatureDescriptor
from cifa_challenge.pipeline.feature_detector import FeatureDetector
from cifa_challenge.progress_bar import ProgressBar
from image_dataset import ImageDataset


def execute_pipeline():
    image_dataset = ImageDataset()

    LABEL_COUNT = len(image_dataset.CIFAR_10_LABELS)
    DATA_BATCH_1 = 'data_batch_1'
    image_contexts = image_dataset.load_training_data(batch=DATA_BATCH_1)[:100]
    test_image_contexts = image_dataset.load_test_data()[:100]

    def _pipeline():
        descriptor_compute_bar = ProgressBar()
        descriptor_compute_bar.prefix = 'Computing descriptor'
        codeBookBar = ProgressBar()
        codeBookBar.prefix = 'Code Book'
        feature_detection = FeatureUnion([("dense_detector", DenseDetector(radiuses=[3, 6, 8, 12, 16], overlap=0.3)),
                                          ("sift_detector", FeatureDetector(detector='sift'))])
        pipeline = Pipeline([("feature_detection", feature_detection),
                             ("grayscale_transform", ColorSpaceTransformer(transformation='grayscale')),
                             ("surf_descriptor", FeatureDescriptor(descriptor_compute_bar, 'surf')),
                             ("code_book", CodeBook(codeBookBar, LABEL_COUNT * 1.5)),
                             ("normalization", StandardScaler(copy=False)),
                             ("dim_reduction", PCA(0.75)),
                             ("classification", svm.SVC(decision_function_shape='ovr', cache_size=2000, verbose=True))])
        # pipeline = Pipeline([("dense_detector", DenseDetector(radiuses=[3, 6, 8, 12, 16], overlap=0.3)),
        #                      ("grayscale_transform",
        #                       ColorSpaceTransformer(transformation='transformed_color_distribution')),
        #                      ("surf_descriptor", FeatureDescriptor(descriptor_extract_bar, 'color-surf')),
        #                      ("code_book", CodeBook(codeBookBar)),
        #                      ("normalization", StandardScaler(copy=False)),
        #                      ("dim_reduction", PCA(0.75)),
        #                      ("classification", svm.SVC(decision_function_shape='ovr', cache_size=2000, verbose=True))])

        pipeline = pipeline.fit(image_contexts, [x.label for x in image_contexts])
        score = pipeline.score(test_image_contexts, [x.label for x in test_image_contexts])
        print('Score: ' + str(score))

        print('Computing confusion matrix')
        plot_confusion_matrix(test_image_contexts, pipeline.predict(test_image_contexts),
                              image_dataset.CIFAR_10_LABELS)

    def _grid_search():
        pass

    def test_1():
        pipeline = Pipeline(
            [("color_space_transformer", ColorSpaceTransformer(transformation='transformed_color_distribution'))])
        images, kps = pipeline.fit_transform((image_contexts, []), None)

        img_grid = [images]
        plot_image_grid(img_grid, (1, len(images)), '../../results/transformed_color_space.png')
        print('Done')

    def test_2():
        ct = ColorSpaceTransformer(transformation='transformed_color_distribution')
        x = (image_contexts, [])
        ct.fit(x).transform(x)

    # test_1()
    _pipeline()


def plot_confusion_matrix(test_image_contexts, predictions, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    # compute confusion matrix
    truth = [test_img.label for test_img in test_image_contexts]
    confusion_matrix = sklearn.metrics.confusion_matrix(truth, predictions, range(len(classes)))

    """
    Print and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(confusion_matrix)

    plt.ioff()
    figure = Figure(figsize=(10, 10))
    canvas = FigureCanvas(figure)

    ax = figure.subplots()
    cax = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    figure.colorbar(cax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    ax.xaxis.tick_top()

    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        ax.text(j, i, format(confusion_matrix[i, j], fmt),
                horizontalalignment="center",
                color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    canvas.print_figure('../../results/confusion_matrix.png')
