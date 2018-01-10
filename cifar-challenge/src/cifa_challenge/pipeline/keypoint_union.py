from sklearn.base import TransformerMixin, BaseEstimator

from cifa_challenge.my_logger import MyLogger
from cifa_challenge.progress_bar import ProgressBar


class KeypointUnion(BaseEstimator, TransformerMixin):
    def __init__(self, progressBar=ProgressBar(), keypoint_detector_list=[]):
        self._logger = MyLogger.getLogger('cifar-challenge.KeypointUnion')
        self._progressBar = progressBar
        self.keypoint_detector_list = keypoint_detector_list

    def fit(self, images, y=None):
        for detector_name, detector in self.keypoint_detector_list:
            self._logger.info('Fitting keypoint detector' + detector_name)
            detector.fit(images)
        return self

    def transform(self, images):
        kps = [[] for i in images]
        for detector_name, detector in self.keypoint_detector_list:
            self._logger.info('Detecting keypoints using ' + detector_name)
            image, detected_kps = detector.transform(images)

            for i, imp_kps in enumerate(detected_kps):
                kps[i].extend(imp_kps)
        return images, kps
