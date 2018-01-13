from sklearn import svm

from cifa_challenge.my_logger import MyLogger


class Classifier:
    def __init__(self):
        self._svm = svm.SVC(decision_function_shape='ovr', cache_size=2000, verbose=True)
        # self._svm = SGDClassifier(decision_function_shape='ovr', cache_size=2000, verbose=True, n_jobs=-1)
        self._imageContexts = None
        self._kdtree = None
        self._logger = MyLogger.getLogger('cifar-challenge.Classifier')

    def learn(self, image_contexts):
        self._imageContexts = image_contexts
        code_vectors = [imageContext.code_vector for imageContext in image_contexts]
        self._logger.info('Training classifier on code_vectors: ' + str((len(code_vectors), len(image_contexts))))
        self._svm.fit(code_vectors,
                      [imageContext.label for imageContext in image_contexts])
        # self._kdtree = KDTree(code_vectors, leaf_size=5)

        return self

    def predict(self, test_image_contexts):
        return self._svm.predict([imageContext.code_vector for imageContext in test_image_contexts])

    def score(self, test_image_contexts):
        return self._svm.score([imageContext.code_vector for imageContext in test_image_contexts],
                               [imageContext.label for imageContext in test_image_contexts])

    #
    # def knn(self, image_context, k):
    #     nn = self._kdtree.query([image_context.code_vector], k=k, return_distance=False)[0]
    #     return self._imageContexts[nn]
