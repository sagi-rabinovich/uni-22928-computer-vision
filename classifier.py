from sklearn import svm


class Classifier:
    def __init__(self):
        self._svm = svm.SVC()

    def learn(self, imageContexts):
        self._svm.fit([imageContext.codeVector for imageContext in imageContexts],
                      [imageContext.label for imageContext in imageContexts])
        return self

    def predict(self, imageContext):
        self._svm.predict(imageContext.codeVector)
        return self

    def score(self, imageContexts):
        self._svm.score([imageContext.codeVector for imageContext in imageContexts],
                        [imageContext.label for imageContext in imageContexts])
        return self
