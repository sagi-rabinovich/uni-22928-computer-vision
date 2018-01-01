from sklearn import svm
from sklearn.neighbors import KDTree


class Classifier:
    def __init__(self):
        self._svm = svm.SVC(decision_function_shape='ovr', cache_size=1000, verbose=True)
        self._imageContexts = None
        self._kdtree = None

    def learn(self, imageContexts):
        self._imageContexts = imageContexts
        codeVectors = [imageContext.codeVector for imageContext in imageContexts]
        self._svm.fit(codeVectors,
                      [imageContext.label for imageContext in imageContexts])
        self._kdtree = KDTree(codeVectors, leaf_size=5)

        return self

    def predict(self, testImageContexts):
        return self._svm.predict([imageContext.codeVector for imageContext in testImageContexts])

    def score(self, testImageContexts):
        return self._svm.score([imageContext.codeVector for imageContext in testImageContexts],
                               [imageContext.label for imageContext in testImageContexts])

    def knn(self, imageContext, k):
        nn = self._kdtree.query([imageContext.codeVector], k=k, return_distance=False)[0]
        return self._imageContexts[nn]
