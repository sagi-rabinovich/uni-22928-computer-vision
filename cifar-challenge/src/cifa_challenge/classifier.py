import itertools

import numpy as np
import sklearn
from matplotlib import pyplot as plt
from matplotlib.backends.backend_template import FigureCanvas
from matplotlib.figure import Figure
from sklearn import svm
from sklearn.neighbors import KDTree


class Classifier:
    def __init__(self):
        self._svm = svm.SVC(decision_function_shape='ovr', cache_size=1000, verbose=True)
        self._imageContexts = None
        self._kdtree = None

    def learn(self, image_contexts):
        self._imageContexts = image_contexts
        code_vectors = [imageContext.code_vector for imageContext in image_contexts]
        self._svm.fit(code_vectors,
                      [imageContext.label for imageContext in image_contexts])
        self._kdtree = KDTree(code_vectors, leaf_size=5)

        return self

    def predict(self, test_image_contexts):
        return self._svm.predict([imageContext.code_vector for imageContext in test_image_contexts])

    def score(self, test_image_contexts):
        return self._svm.score([imageContext.code_vector for imageContext in test_image_contexts],
                               [imageContext.label for imageContext in test_image_contexts])

    def knn(self, image_context, k):
        nn = self._kdtree.query([image_context.code_vector], k=k, return_distance=False)[0]
        return self._imageContexts[nn]

    def plot_confusion_matrix(self, test_image_contexts, predictions, classes,
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
