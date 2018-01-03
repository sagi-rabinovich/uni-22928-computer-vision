import cv2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_template import FigureCanvas
from matplotlib.figure import Figure

from dense_detector import DenseDetector
from image_dataset import ImageDataset
from progress_bar import ProgressBar

progressBar = ProgressBar()
imageDataset = ImageDataset()

nimages = 30
imgs = imageDataset.load_training_data(samples=nimages)

detectors = [
    ('Surf', cv2.xfeatures2d.SURF_create()),
    ('Sift', cv2.xfeatures2d.SIFT_create()),
    # ('MSER', cv2.MSER_create()),
    # ('ORB', cv2.ORB_create()),
    # ('Kaze', cv2.KAZE_create()),
    ('AKaze', cv2.AKAZE_create()),
    ('Dense_8', DenseDetector([8], 0.3)),
    ('Dense_12', DenseDetector([12], 0.3))
]

plt.ioff()
figure = Figure(figsize=(10, 10))
canvas = FigureCanvas(figure)


def drawKeyPoints(original, gray, kp):
    outImg = original.copy()
    cv2.drawKeypoints(gray, kp, outImg, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return outImg


cols = len(detectors) + 2
for im_i in range(len(imgs)):
    im = imgs[im_i]

    ax = figure.add_subplot(nimages, cols, (im_i * cols) + 1)
    ax.set_axis_off()
    ax.imshow(im.original, interpolation='nearest')

    ax = figure.add_subplot(nimages, cols, (im_i * cols) + 2)
    ax.set_axis_off()
    ax.imshow(im.gray, interpolation='nearest')

    for detector_idx in range(len(detectors)):
        detector = detectors[detector_idx][1]
        detector_name = detectors[detector_idx][0]
        kp = detector.detect(im.gray, None)

        im_with_kp = drawKeyPoints(im.original, im.gray, kp)

        ax = figure.add_subplot(nimages, cols, (im_i * cols) + detector_idx + 3)
        ax.set_axis_off()
        ax.imshow(im_with_kp, interpolation='nearest')
        if im_i == 0:
            ax.set_title(detector_name)

canvas.print_figure('../../results/feature_extraction_compare.png')
