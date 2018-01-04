import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class ColorSpaceTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, transformation='transformed_color_distribution'):
        self.transformation = transformation
        self._transformer = None

    def fit(self, images, y=None):
        if self.transformation == 'transformed_color_distribution':
            self._transformer = self._transformed_color_distribution
        elif self.transformation == 'grayscale':
            self._transformer = self._grayscale
        else:
            raise RuntimeError('Unknown transformation [' + str(self.transformation) + ']')

        return self

    def transform(self, images_with_kps):
        images = images_with_kps[0]
        kps = images_with_kps[1]
        return self._transformer(images), kps

    def _grayscale(self, images):
        return [x.gray for x in images]

    def _transformed_color_distribution(self, images):
        color_images = [x.original for x in images]
        axis = (1, 2)
        mean = np.mean(color_images, axis=axis)
        std = np.std(color_images, axis=axis)
        old_err_state = np.seterr(divide='raise')
        result = []
        for i, color_img in enumerate(color_images):
            m = mean[i]
            s = std[i]
            normalized_img = np.divide(np.subtract(color_img, m), s)
            max_c = np.max(normalized_img, (0, 1))
            min_c = np.min(normalized_img, (0, 1))
            normalized_img = np.multiply(np.divide(np.subtract(normalized_img, min_c), np.subtract(max_c, min_c)),
                                         255).astype('uint8')
            result.append(normalized_img)
        np.seterr(**old_err_state)
        return result
