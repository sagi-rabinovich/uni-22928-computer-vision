import sys

import cv2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_template import FigureCanvas
from matplotlib.figure import Figure

from image_dataset import ImageDataset
from progress_bar import ProgressBar

progressBar = ProgressBar()
imageDataset = ImageDataset()

nimages = 30
imgs = imageDataset.load_training_data(batches='data_batch_1')[:nimages]
detectors = [
    ('Sift', cv2.xfeatures2d.SIFT_create()),
    ('Surf', cv2.xfeatures2d.SURF_create(hessianThreshold=300)),
    ('Star', cv2.xfeatures2d.StarDetector_create()),
    ('Brisk', cv2.BRISK_create()),
    ('MSER', cv2.MSER_create()),
    ('Kaze', cv2.KAZE_create()),
    # ('AKaze', cv2.AKAZE_create())
]

plt.ioff()
figure = Figure(figsize=(30, 30))
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
        try:
            img_eq = im.gray  # cv2.equalizeHist(im.gray)
            kp = detector.detect(img_eq, None)
            im_with_kp = drawKeyPoints(im.original, img_eq, kp)
            ax = figure.add_subplot(nimages, cols, (im_i * cols) + detector_idx + 3)
            ax.set_axis_off()
            ax.imshow(im_with_kp, interpolation='nearest')
            if im_i == 0:
                ax.set_title(detector_name)
        except:
            print "Unexpected error with detector: ", detector_name, sys.exc_info()[0]

canvas.print_figure('../results/feature_extraction_compare.png')
