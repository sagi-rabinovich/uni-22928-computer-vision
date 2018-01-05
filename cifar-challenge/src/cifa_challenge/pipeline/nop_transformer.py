from sklearn.base import TransformerMixin, BaseEstimator


class NopTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return X
