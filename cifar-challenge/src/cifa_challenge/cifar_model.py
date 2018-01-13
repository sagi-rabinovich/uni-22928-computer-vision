import itertools

import numpy as np
import sklearn
from matplotlib import pyplot as plt
from matplotlib.backends.backend_template import FigureCanvas
from matplotlib.figure import Figure
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, FunctionTransformer

from cifa_challenge.my_logger import MyLogger
from cifa_challenge.pipeline.bilateral_filter import BilateralFilter
from cifa_challenge.pipeline.hard_code_book import HardCodeBook
from cifa_challenge.pipeline.soft_code_book import SoftCodeBook
from image_dataset import ImageDataset
from pipeline.color_space_transformer import ColorSpaceTransformer
from pipeline.dense_detector import DenseDetector
from pipeline.feature_descriptor import FeatureDescriptor
from pipeline.feature_detector import FeatureDetector
from progress_bar import ProgressBar


class CifarModel():
    """
    This holds the implementation of my classifier for the CIFAR-10 Challenge.
    The method train
    """

    def __init__(self, dataset_dir=None):
        self.image_dataset_ = ImageDataset(dataset_dir)

        self.LABEL_COUNT = len(self.image_dataset_.CIFAR_10_LABELS)
        DATA_BATCH = ['data_batch_1']
        samples = -1

        self.image_contexts_ = self.image_dataset_.load_training_data(batches=DATA_BATCH, samples=samples)
        self.test_image_contexts_ = self.image_dataset_.load_test_data(samples=samples)
        self.logger_ = MyLogger.getLogger('cifar-challenge.Pipeline')

        self.keypoint_detector_bar_ = ProgressBar()
        self.keypoint_detector_bar_.prefix = 'Detecting keypoints'
        self.descriptor_compute_bar_ = ProgressBar()
        self.descriptor_compute_bar_.prefix = 'Computing descriptor'
        self.codebook_bar_ = ProgressBar()
        self.codebook_bar_.prefix = 'Code Book'

    def load_pipeline(self, filename):
        """
        load a trained pipeline (classifier) from a pickle file
        :param filename:
        :return:
        """
        pipeline = joblib.load(filename)
        return pipeline

    def dump_pipeline(self, pipeline, filename):
        """
        save a train pipeline to pickle file
        :param pipeline:
        :param filename:
        :return:
        """
        dense_detector = pipeline.named_steps.get('dense_detector')
        if dense_detector:
            pipeline.named_steps.dense_detector.prepare_to_pickle()
        joblib.dump(pipeline, filename)

    @staticmethod
    def normalize_image_descriptors(X, y=None):
        """
        normalize image features using L2-norm. This method is specific to the pipeline implementation
        :param X: a tuple of image contexts and descriptors
        :param y:
        :return:
        """
        descriptors = []
        normalizer = Normalizer(copy=False)
        for desc in X[1]:
            descriptors.append(normalizer.transform(desc))
        return X[0], descriptors

    def __extractLabels(self, imgs):
        return [x.label for x in imgs]

    def __scoreAndPrint(self, pipeline):
        self.logger_.info('Computing score')
        score = pipeline.score(self.test_image_contexts_, self.__extractLabels(self.test_image_contexts_))
        print('Score: ' + str(score))

    def __best_model(self):
        """
        :return: The pipeline representing the most accurate model I could achieve
        """
        return Pipeline(
            [("color_transform", ColorSpaceTransformer(transformation='transformed_color_distribution')),
             ("smoothing", BilateralFilter()),
             ("dense_detector", DenseDetector(radiuses=[2, 4, 8, 16], overlap=0.3)),
             ("surf_descriptor", FeatureDescriptor(self.descriptor_compute_bar_, 'color-sift')),
             ("descriptor_nomalizer", FunctionTransformer(CifarModel.normalize_image_descriptors, validate=False)),
             ("code_book", SoftCodeBook(self.codebook_bar_, self.LABEL_COUNT * 3)),
             ("scaler", StandardScaler(copy=False)),
             ("classification",
              svm.SVC(C=300, gamma=0.00001, decision_function_shape='ovr', cache_size=2000, verbose=True))])

    def __naive_model(self):
        return Pipeline([("grayscale_transform", ColorSpaceTransformer(transformation='grayscale')),
                         ("detector", FeatureDetector(detector='sift')),
                         ("descriptor", FeatureDescriptor(self.descriptor_compute_bar_, 'sift')),
                         ("code_book", HardCodeBook(self.codebook_bar_, vocabulary_size=2000)),
                         ("classification", svm.SVC(decision_function_shape='ovr', cache_size=2000, verbose=True))])

    def train_best_pipeline(self):
        """
        Trains the best model on the configured set of images, computes its score and plots its confusion matrix
        :return: None
        """
        pipeline = self.__best_model()
        pipeline = pipeline.fit(self.image_contexts_, self.__extractLabels(self.image_contexts_))
        self.__scoreAndPrint(pipeline)
        # dump_pipeline(pipeline, 'best_score.pkl')

        self.logger_.info('Making predictions')
        predictions = pipeline.predict(self.test_image_contexts_)
        self.logger_.info('Computing confusion matrix')
        self.plot_confusion_matrix(predictions, filename_suffix='some_pipe_line')

    def plot_confusion_matrix(self,
                              predictions,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues,
                              filename_suffix=None):
        # compute confusion matrix
        truth = [test_img.label for test_img in self.test_image_contexts_]
        categories = self.image_dataset_.CIFAR_10_LABELS
        confusion_matrix = sklearn.metrics.confusion_matrix(truth, predictions, range(len(categories)))

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
        tick_marks = np.arange(len(categories))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(categories)
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

        canvas.print_figure('../../results/cm_' + filename_suffix + '_.png')

    def _multiple_pipelines(self):
        '''
        Used to compare the performance of the various improvements applied to the model
        :return:
        '''
        pipes = [
            ("sift-sift",
             Pipeline([("grayscale_transform", ColorSpaceTransformer(transformation='grayscale')),
                       ("smoothing", BilateralFilter()),
                       ("detector", FeatureDetector(detector='sift')),
                       ("descriptor", FeatureDescriptor(self.descriptor_compute_bar_, 'sift')),
                       ("code_book", HardCodeBook(self.codebook_bar_, vocabulary_size=4000)),
                       ("classification",
                        svm.SVC(decision_function_shape='ovr', cache_size=2000, verbose=True))])),
            ("dense-sift",
             Pipeline([("grayscale_transform", ColorSpaceTransformer(transformation='grayscale')),
                       ("smoothing", BilateralFilter()),
                       ("detector", DenseDetector(radiuses=[3, 6, 12, 16], overlap=0.3)),
                       ("descriptor", FeatureDescriptor(self.descriptor_compute_bar_, 'sift')),
                       ("code_book", HardCodeBook(self.codebook_bar_, vocabulary_size=4000)),
                       ("classification",
                        svm.SVC(decision_function_shape='ovr', cache_size=2000, verbose=True))])),
            ("image_smoothing_dense",
             Pipeline([("grayscale_transform", ColorSpaceTransformer(transformation='grayscale')),
                       ("smoothing", BilateralFilter()),
                       ("detector", DenseDetector(radiuses=[3, 6, 12, 16], overlap=0.3)),
                       ("descriptor", FeatureDescriptor(self.descriptor_compute_bar_, 'sift')),
                       ("code_book", HardCodeBook(self.codebook_bar_, vocabulary_size=4000)),
                       ("classification",
                        svm.SVC(decision_function_shape='ovr', cache_size=2000, verbose=True))])),

            ("color_descriptor",
             Pipeline([("color_transform", ColorSpaceTransformer(transformation='transformed_color_distribution')),
                       ("smoothing", BilateralFilter()),
                       ("detector", DenseDetector(radiuses=[3, 6, 12, 16], overlap=0.3)),
                       ("descriptor", FeatureDescriptor(self.descriptor_compute_bar_, 'color-sift')),
                       ("code_book", HardCodeBook(self.codebook_bar_, vocabulary_size=4000)),
                       ("classification",
                        svm.SVC(decision_function_shape='ovr', cache_size=2000, verbose=True))])),
            ("color_descriptor_no_smoothing",
             Pipeline([("color_transform", ColorSpaceTransformer(transformation='transformed_color_distribution')),
                       ("detector", DenseDetector(radiuses=[3, 6, 12, 16], overlap=0.3)),
                       ("descriptor", FeatureDescriptor(self.descriptor_compute_bar_, 'color-sift')),
                       ("code_book", HardCodeBook(self.codebook_bar_, vocabulary_size=4000)),
                       ("classification",
                        svm.SVC(decision_function_shape='ovr', cache_size=2000, verbose=True))])),

            ("codeword_scaler",
             Pipeline([("color_transform", ColorSpaceTransformer(transformation='transformed_color_distribution')),
                       ("smoothing", BilateralFilter()),
                       ("detector", DenseDetector(radiuses=[3, 6, 12, 16], overlap=0.3)),
                       ("descriptor", FeatureDescriptor(self.descriptor_compute_bar_, 'color-sift')),
                       ("code_book", HardCodeBook(self.codebook_bar_, vocabulary_size=4000)),
                       ("scaler", StandardScaler(copy=False)),
                       ("classification",
                        svm.SVC(decision_function_shape='ovr', cache_size=2000, verbose=True))])),

            ("soft_code_book",
             Pipeline([("color_transform", ColorSpaceTransformer(transformation='transformed_color_distribution')),
                       ("smoothing", BilateralFilter()),
                       ("detector", DenseDetector(radiuses=[3, 6, 12, 16], overlap=0.3)),
                       ("descriptor", FeatureDescriptor(self.descriptor_compute_bar_, 'color-sift')),
                       ("code_book", SoftCodeBook(self.codebook_bar_, vocabulary_size_factor=self.LABEL_COUNT * 2)),
                       ("scaler", StandardScaler(copy=False)),
                       ("classification",
                        svm.SVC(decision_function_shape='ovr', cache_size=2000, verbose=True))])),

            ("normalize_before_soft_code_book",
             Pipeline([("color_transform", ColorSpaceTransformer(transformation='transformed_color_distribution')),
                       ("smoothing", BilateralFilter()),
                       ("detector", DenseDetector(radiuses=[3, 6, 12, 16], overlap=0.3)),
                       ("descriptor", FeatureDescriptor(self.descriptor_compute_bar_, 'color-sift')),
                       ("descriptor_nomalizer", FunctionTransformer(self.normalize_image_descriptors, validate=False)),
                       ("code_book", SoftCodeBook(self.codebook_bar_, vocabulary_size_factor=self.LABEL_COUNT * 2)),
                       ("scaler", StandardScaler(copy=False)),
                       ("classification", svm.SVC(decision_function_shape='ovr', cache_size=2000, verbose=True))]))]

        scores = []
        for pipe_name, pipe in pipes:
            self.logger_.info('Computing pipe: ' + pipe_name)
            pipe.fit(self.image_contexts_, self.__extractLabels(self.image_contexts_))
            self.logger_.info('Computing score: ' + pipe_name)
            score = pipe.score(self.test_image_contexts_, self.__extractLabels(self.test_image_contexts_))
            self.logger_.info(pipe_name + ' - Score: ' + str(score))
            scores.append((pipe_name, score))

            self.logger_.info(pipe_name + ' - Making predictions')
            predict = pipe.predict(self.test_image_contexts_)
            self.logger_.info(pipe_name + ' - Computing confusion matrix')
            self.plot_confusion_matrix(self.test_image_contexts_, predict,
                                       self.image_dataset_.CIFAR_10_LABELS, filename_suffix=pipe_name)

        print('final scores:')
        for pipe_name, score in scores:
            print('%s - %s' % (pipe_name, score))

    def _grid_search(self):
        '''
        Used to perform a grid search over parameter space to optimize the model's parameters.
        :return:
        '''
        pipeline = Pipeline(
            [("color_transform", ColorSpaceTransformer(transformation='transformed_color_distribution'),),
             #   ("smoothing", BilateralFilter()),
             ("dense_detector", DenseDetector(radiuses=[2, 4, 8, 16], overlap=0.3)),
             ("surf_descriptor", FeatureDescriptor(ProgressBar(), descriptor='color-sift')),
             ("code_book", SoftCodeBook(vocabulary_size_factor=self.LABEL_COUNT * 3, sparse_transform_alpha=0)),
             ("normalization", StandardScaler(copy=False)),
             #       ("dim_reduction", PCA(n_components=0.75, whiten=True)),
             ("classification", svm.SVC(decision_function_shape='ovr', cache_size=2000, C=300, gamma=0.00001))])

        param_grid = {
            # 'dense_detector__overlap': [0, 0.3],
            'dense_detector__radiuses': [[2, 4, 8, 16]],
            'code_book__vocabulary_size_factor': [self.LABEL_COUNT * 3],
            'code_book__sparse_transform_alpha': [0, 0.1, 0.3, 0.5],
            # 'dim_reduction': [PCA(0.85, whiten=True)],
            # 'dim_reduction__n_components': [0.75, 0.85, 0.9],
            # 'classification__C': [300, 500],
            'classification__gamma': [0.00001, 0.000005]
        }

        gridSearch = GridSearchCV(pipeline, param_grid, n_jobs=4, verbose=10, refit=False, error_score=0, cv=2)
        gridSearch.fit(self.image_contexts_, self.__extractLabels(self.image_contexts_))
        print('\n\n===================\nGrid search results:')
        print(gridSearch.best_params_)
        print("score:" + str(gridSearch.best_score_))
