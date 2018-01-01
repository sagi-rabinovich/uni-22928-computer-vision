from sklearn import svm


class Classifier:
    def __init__(self):
        self._svm = svm.SVC()

    def learn(self, imageContexts):
        self._svm.fit([imageContext.codeVector for imageContext in imageContexts],
                      [imageContext.label for imageContext in imageContexts])
        return self

    def predict(self, imageContexts):
        return self._svm.predict([imageContext.codeVector for imageContext in imageContexts])

    def score(self, testImageContexts):
        return self._svm.score([imageContext.codeVector for imageContext in testImageContexts],
                               [imageContext.label for imageContext in testImageContexts])
