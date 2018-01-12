import itertools

import cv2
import numpy as np
import sklearn
from matplotlib import pyplot as plt
from matplotlib.backends.backend_template import FigureCanvas
from matplotlib.figure import Figure
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer, Normalizer

from cifa_challenge.my_logger import MyLogger
from cifa_challenge.pipeline.bilateral_filter import BilateralFilter
from cifa_challenge.pipeline.keypoint_union import KeypointUnion
from image_dataset import ImageDataset
from pipeline.code_book import CodeBook
from pipeline.color_space_transformer import ColorSpaceTransformer
from pipeline.dense_detector import DenseDetector
from pipeline.feature_descriptor import FeatureDescriptor
from pipeline.feature_detector import FeatureDetector
from pipeline.nop_transformer import NopTransformer
from progress_bar import ProgressBar


def execute_pipeline():
    image_dataset = ImageDataset()

    LABEL_COUNT = len(image_dataset.CIFAR_10_LABELS)
    DATA_BATCH = 'data_batch_1'
    samples = -1

    image_contexts = image_dataset.load_training_data(batch=DATA_BATCH, samples=samples)
    test_image_contexts = image_dataset.load_test_data(samples=samples)
    logger = MyLogger.getLogger('cifar-challenge.Pipeline')

    def extractLabels(imgs, repeat):
        labels = []
        for i in range(repeat):
            for x in imgs:
                labels.append(x.label)
        return labels

    def _pipeline():
        keypoint_detector_bar = ProgressBar()
        keypoint_detector_bar.prefix = 'Detecting keypoints'
        descriptor_compute_bar = ProgressBar()
        descriptor_compute_bar.prefix = 'Computing descriptor'
        codeBookBar = ProgressBar()
        codeBookBar.prefix = 'Code Book'

        def multi_descriptor_pipeline():
            kaze_pipeline = Pipeline([("grayscale_transform", ColorSpaceTransformer(transformation='grayscale')),
                                      ("kaze_detector", FeatureDetector(detector='kaze')),
                                      ("surf_descriptor", FeatureDescriptor(descriptor_compute_bar, 'kaze')),
                                      ("code_book", CodeBook(LABEL_COUNT * 1.5))])
            sift_pipeline = Pipeline([("grayscale_transform", ColorSpaceTransformer(transformation='grayscale')),
                                      ("sift_detector", FeatureDetector(detector='sift')),
                                      ("sift_descriptor", FeatureDescriptor(descriptor_compute_bar, 'sift')),
                                      ("code_book", CodeBook(LABEL_COUNT * 1.5))])
            surf_pipeline = Pipeline([("grayscale_transform", ColorSpaceTransformer(transformation='grayscale')),
                                      ("surf_detector", FeatureDetector(detector='surf')),
                                      ("surf_descriptor", FeatureDescriptor(descriptor_compute_bar, 'surf')),
                                      ("code_book", CodeBook(LABEL_COUNT * 1.5))])
            dense_pipeline = Pipeline([("grayscale_transform", ColorSpaceTransformer(transformation='grayscale')),
                                       ("dense_detector", DenseDetector(radiuses=[3, 6, 8, 12, 16], overlap=0.3)),
                                       ("surf_descriptor", FeatureDescriptor(descriptor_compute_bar, 'surf')),
                                       ("code_book", CodeBook(LABEL_COUNT * 1.5))])

            pipeline = Pipeline([
                ("vectorization", FeatureUnion([
                    ("kaze_pipeline", kaze_pipeline),
                    ("sift_pipeline", sift_pipeline),
                    ("surf_pipeline", surf_pipeline),
                    ("dense_pipeline", dense_pipeline)])),
                ("debug", NopTransformer()),
                ("normalization", StandardScaler(copy=False)),
                ("dim_reduction", PCA(0.8)),
                ("classification", svm.SVC(decision_function_shape='ovr', cache_size=2000, verbose=True))])
            return pipeline

        def color_multi_descriptor_pipeline():
            kaze_pipeline = Pipeline([("color_transform", ColorSpaceTransformer(transformation='grayscale')),
                                      ("kaze_detector", FeatureDetector(detector='kaze')),
                                      ("surf_descriptor", FeatureDescriptor(descriptor_compute_bar, 'kaze')),
                                      ("code_book", CodeBook(LABEL_COUNT * 1.5))])
            sift_pipeline = Pipeline(
                [("color_transform", ColorSpaceTransformer(transformation='transformed_color_distribution')),
                 ("sift_detector", FeatureDetector(detector='sift')),
                 ("sift_descriptor", FeatureDescriptor(descriptor_compute_bar, 'color-sift')),
                 ("code_book", CodeBook(LABEL_COUNT * 1.5))])
            surf_pipeline = Pipeline(
                [("color_transform", ColorSpaceTransformer(transformation='transformed_color_distribution')),
                 ("surf_detector", FeatureDetector(detector='surf')),
                 ("surf_descriptor", FeatureDescriptor(descriptor_compute_bar, 'color-surf')),
                 ("code_book", CodeBook(LABEL_COUNT * 1.5))])
            dense_pipeline = Pipeline(
                [("color_transform", ColorSpaceTransformer(transformation='transformed_color_distribution')),
                 ("dense_detector", DenseDetector(radiuses=[3, 6, 8, 12, 16], overlap=0.3)),
                 ("surf_descriptor", FeatureDescriptor(descriptor_compute_bar, 'color-surf')),
                 ("code_book", CodeBook(LABEL_COUNT * 1.5))])

            pipeline = Pipeline([
                ("vectorization", FeatureUnion([
                    ("kaze_pipeline", kaze_pipeline),
                    ("sift_pipeline", sift_pipeline),
                    ("surf_pipeline", surf_pipeline),
                    ("dense_pipeline", dense_pipeline)])),
                ("debug", NopTransformer()),
                ("normalization", StandardScaler(copy=False)),
                ("dim_reduction", PCA(0.8)),
                ("classification", svm.SVC(decision_function_shape='ovr', cache_size=2000, verbose=True))])
            return pipeline

        def dense_descriptor_pipeline():
            return Pipeline([("grayscale_transform", ColorSpaceTransformer(transformation='grayscale')),
                             ("smoothing", FunctionTransformer(lambda X: cv2.bilateralFilter(X.original, 4, 100, 100))),
                             ("dense_detector", DenseDetector(radiuses=[3, 6, 8, 12, 16], overlap=0.3)),
                             ("surf_descriptor", FeatureDescriptor(descriptor_compute_bar, 'surf')),
                             ("code_book", CodeBook(LABEL_COUNT * 1.5)),
                             ("normalization", StandardScaler(copy=False)),
                             ("dim_reduction", PCA(0.75)),
                             ("classification", svm.SVC(decision_function_shape='ovr', cache_size=2000, verbose=True))])

        def color_dense_descriptor_pipeline():
            def power_transform(X, alpha):
                return np.multiply(np.sign(X), np.power(np.abs(X), 2))

            return Pipeline(
                [("color_transform", ColorSpaceTransformer(transformation='transformed_color_distribution')),
                 #   ("smoothing", BilateralFilter()),
                 # ("dense_detector", DenseDetector(radiuses=[3, 6, 8, 12], overlap=0.3)),
                 ("keypoint_detectors", KeypointUnion(ProgressBar(), [
                     ('dense_detector', DenseDetector(radiuses=[3, 6], overlap=0.3)),
                     ("sift_detector", FeatureDetector(progressBar=keypoint_detector_bar, detector='sift')),
                     ("surf_detector", FeatureDetector(progressBar=keypoint_detector_bar, detector='surf'))]
                                                      )),
                 ("surf_descriptor", FeatureDescriptor(descriptor_compute_bar, 'color-surf')),
                 ("code_book", CodeBook(LABEL_COUNT * 3)),
                 ("l2_normalization", Normalizer(norm='l2', copy=False)),
                 # ("power_normalization", FunctionTransformer(power_transform, kw_args={'alpha': 0.5})),
                 ("normalization", StandardScaler(copy=False)),
                 ("dim_reduction", PCA(0.75)),
                 ("classification", svm.SVC(decision_function_shape='ovr', cache_size=2000, verbose=True))])

        # pipeline = Pipeline([("feature_extraction", FeatureUnion([
        #     ("sift_surf", Pipeline([("sift_detector", FeatureDetector(detector='sift')),
        #                             ("grayscale_transform", ColorSpaceTransformer(transformation='grayscale')),
        #                             ("surf_descriptor", FeatureDescriptor(descriptor_compute_bar, 'surf'))])),
        #     ("surf_surf", Pipeline([("surf_detector", FeatureDetector(detector='sift')),
        #                             ("grayscale_transform", ColorSpaceTransformer(transformation='grayscale')),
        #                             ("surf_descriptor", FeatureDescriptor(descriptor_compute_bar, 'surf'))]))])),
        #                      ("code_book", CodeBook(LABEL_COUNT * 1.5)),
        #                      ("normalization", StandardScaler(copy=False)),
        #                      ("dim_reduction", PCA(0.75)),
        #                      ("classification", svm.SVC(decision_function_shape='ovr', cache_size=2000, verbose=True))])

        # pipeline = Pipeline([("dense_detector", DenseDetector(radiuses=[3, 6, 8, 12, 16], overlap=0.3)),
        #                      ("grayscale_transform",
        #                       ColorSpaceTransformer(transformation='transformed_color_distribution')),
        #                      ("surf_descriptor", FeatureDescriptor(descriptor_extract_bar, 'color-surf')),
        #                      ("code_book", CodeBook(codeBookBar)),
        #                      ("normalization", StandardScaler(copy=False)),
        #                      ("dim_reduction", PCA(0.75)),
        #                      ("classification", svm.SVC(decision_function_shape='ovr', cache_size=2000, verbose=True))])
        def testing():
            kaze_pipeline = Pipeline(
                [("grayscale_transform", ColorSpaceTransformer(transformation='transformed_color_distribution')),
                 ("smoothing", BilateralFilter()),
                 ("kaze_detector", FeatureDetector(detector='kaze')),
                 ("surf_descriptor", FeatureDescriptor(descriptor_compute_bar, 'color-sift')),
                 ("code_book", CodeBook(codeBookBar, vocabulary_size_factor=LABEL_COUNT * 3))])
            sift_pipeline = Pipeline(
                [("grayscale_transform", ColorSpaceTransformer(transformation='transformed_color_distribution')),
                 ("smoothing", BilateralFilter()),
                 ("sift_detector", FeatureDetector(detector='sift')),
                 ("sift_descriptor", FeatureDescriptor(descriptor_compute_bar, 'color-sift')),
                 ("code_book", CodeBook(codeBookBar, vocabulary_size_factor=LABEL_COUNT * 3))])
            return Pipeline(
                [("feature_extraction",
                  FeatureUnion([("kaze_pipeline", kaze_pipeline), ("sift_pipeline", sift_pipeline)])),
                 # ("l2_normalization", Normalizer(norm='l2', copy=False)),
                 ("scaler", StandardScaler(copy=False)),
                 # ("dim_reduction", PCA(0.85, whiten=True)),
                 ("classification",
                  svm.SVC(C=300, gamma=0.00001, decision_function_shape='ovr', cache_size=2000, verbose=True))])

        def best_pipeline():
            return Pipeline(  # this give 0.492
                [("color_transform", ColorSpaceTransformer(transformation='transformed_color_distribution')),
                 ("smoothing", BilateralFilter()),
                 ("dense_detector", DenseDetector(radiuses=[2, 4, 8, 16], overlap=0.3)),
                 ("surf_descriptor", FeatureDescriptor(descriptor_compute_bar, 'color-sift')),
                 ("code_book", CodeBook(codeBookBar, LABEL_COUNT * 3)),
                 ("scaler", StandardScaler(copy=False)),
                 #    ("dim_reduction", PCA(whiten=True)),
                 ("classification",
                  svm.SVC(C=300, gamma=0.00001, decision_function_shape='ovr', cache_size=2000, verbose=True))])

        pipeline = best_pipeline()

        pipeline = pipeline.fit(image_contexts, extractLabels(image_contexts, 1))
        logger.info('Computing score')
        score = pipeline.score(test_image_contexts, extractLabels(test_image_contexts, 1))
        print('Score: ' + str(score))

        logger.info('Making predictions')
        predict = pipeline.predict(test_image_contexts)
        logger.info('Computing confusion matrix')
        plot_confusion_matrix(test_image_contexts, predict,
                              image_dataset.CIFAR_10_LABELS, suffix='dense_colorsift')

    def _grid_search():
        pipeline = Pipeline(
            [("color_transform", ColorSpaceTransformer(transformation='transformed_color_distribution'),),
             #   ("smoothing", BilateralFilter()),
             ("dense_detector", DenseDetector(radiuses=[3, 6, 12, 16], overlap=0.3)),
             ("surf_descriptor", FeatureDescriptor(ProgressBar(), descriptor='color-surf')),
             ("code_book", CodeBook(vocabulary_size_factor=LABEL_COUNT * 3)),
             ("normalization", StandardScaler(copy=False)),
             #       ("dim_reduction", PCA(n_components=0.75, whiten=True)),
             ("classification", svm.SVC(decision_function_shape='ovr', cache_size=2000, C=300, gamma=0.00001))])

        param_grid = {
            # 'dense_detector__overlap': [0, 0.3],
            'dense_detector__radiuses': [[2, 4, 8, 16]],
            'code_book__vocabulary_size_factor': [LABEL_COUNT * 3],
            'dim_reduction': [PCA(0.85, whiten=True)],
            # 'dim_reduction__n_components': [0.75, 0.85, 0.9],
            # 'classification__C': [300, 500],
            'classification__gamma': [0.00001, 0.000005]
        }

        gridSearch = GridSearchCV(pipeline, param_grid, n_jobs=1, verbose=10, refit=False, error_score=0, cv=2)
        gridSearch.fit(image_contexts, extractLabels(image_contexts, 1))
        print('\n\n===================\nGrid search results:')
        print(gridSearch.best_params_)
        print("score:" + str(gridSearch.best_score_))

    def _my_grid_search():
        pipeline = Pipeline(
            [("color_transform", ColorSpaceTransformer(transformation='transformed_color_distribution')),
             #   ("smoothing", BilateralFilter()),
             # ("dense_detector", DenseDetector(radiuses=[3, 6, 8, 12], overlap=0.3)),
             ("keypoint_detectors", KeypointUnion(ProgressBar(), [
                 ('dense_detector', DenseDetector(radiuses=[3, 6, 8, 16], overlap=0.3))]
                                                  # ("sift_detector", FeatureDetector(detector='sift')),
                                                  # ("surf_detector", FeatureDetector(detector='surf'))]
                                                  )),
             ("surf_descriptor", FeatureDescriptor(ProgressBar(), descriptor='color-surf')),
             ("code_book", CodeBook(vocabulary_size_factor=LABEL_COUNT * 3)),
             ("l2_normalization", Normalizer(norm='l2', copy=False)),
             # ("power_normalization", FunctionTransformer(power_transform, kw_args={'alpha': 0.5})),
             ("normalization", StandardScaler(copy=False)),
             ("dim_reduction", PCA(n_components=0.85))])
        truth = extractLabels(image_contexts, 1)
        pipeline = pipeline.fit(image_contexts, truth)
        features = pipeline.transform(image_contexts)
        test_features = pipeline.transform(test_image_contexts)

        param_grid = {
            # 'keypoint_detectors__dense_detector__overlap': [0, 0.3],
            #     'code_book__vocabulary_size_factor':[LABEL_COUNT * 1.7, LABEL_COUNT * 2, LABEL_COUNT * 2.5, LABEL_COUNT * 3]
            #     'dim_reduction__n_components': [0.75, 0.8, 0.85, 0.9],
            'c': [200, 300],
            'gamma': [0.00001, 0.000001, 0.0000001]}
        scores = []
        for c in param_grid['c']:
            for gamma in param_grid['gamma']:
                classPipeline = Pipeline([("classification",
                                           svm.SVC(decision_function_shape='ovr', C=c, gamma=gamma, cache_size=2000,
                                                   verbose=False))])
                score = classPipeline.fit(features, truth).score(test_features, extractLabels(test_image_contexts, 1))
                score = 'c:' + str(c) + '\t\tgamma:' + str(gamma) + '\t\tscore:' + str(score)
                print('@@ score: ' + score)
                scores.append(score)
        for score in scores:
            logger.info(score)

    # _grid_search()

    _pipeline()
    # _my_grid_search()


def plot_confusion_matrix(test_image_contexts, predictions, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, suffix=None):
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

    canvas.print_figure('../../results/cm_' + suffix + '_.png')

# return Pipeline( # this give 0.492
#     [("color_transform", ColorSpaceTransformer(transformation='transformed_color_distribution')),
#      ("smoothing", BilateralFilter()),
#      ("dense_detector", DenseDetector(radiuses=[2, 4, 8, 16], overlap=0.3)),
#
#      # ("sift_detector", FeatureDetector(progressBar=keypoint_detector_bar, detector='sift')),
#      ("surf_descriptor", FeatureDescriptor(descriptor_compute_bar, 'color-sift')),
#      ("code_book", CodeBook(codeBookBar, LABEL_COUNT * 3)),
#
#      # ("l2_normalization", Normalizer(norm='l2', copy=False)),
#      ("scaler", StandardScaler(copy=False)),
#      # ("dim_reduction", PCA(0.85, whiten=True)),
#      ("classification",
#       svm.SVC(C=300, gamma=0.00001, decision_function_shape='ovr', cache_size=2000, verbose=True))])
