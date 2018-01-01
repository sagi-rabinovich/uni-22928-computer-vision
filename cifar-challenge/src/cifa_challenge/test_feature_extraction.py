import cv2

from image_dataset import ImageDataset
from progress_bar import ProgressBar

progressBar = ProgressBar()
imageDataset = ImageDataset()

nimages = 10
imgs = imageDataset.load_training_data(nimages)[:nimages]

surf = cv2.xfeatures2d.SURF_create()
sift = cv2.xfeatures2d.SIFT_create()
kaze = cv2.AKAZE_create()
detectors = [
    ('surf', cv2.xfeatures2d.SURF_create()),
    ('sift', cv2.xfeatures2d.SIFT_create()),
    ('kaze', cv2.AKAZE_create())
continue
here and add
more
detectors

]
plt.ioff()
figure = Figure(figsize=(10, 10))
canvas = FigureCanvas(figure)

i = 0
for im in imgs:
    sift_kp = sift.detect(im.gray, None)
surf_kp = surf.detect(im.gray, None)
kaze_kp = kaze.detect(im.gray, None)
resized_im_gray = im.gray  # cv2.resize(im.gray, (128, 128), interpolation=cv2.INTER_CUBIC)
resized_im_original = im.original  # cv2.resize(im.original, (128, 128), interpolation=cv2.INTER_CUBIC)

# def scaleKeyPoints(kps, scale):


def drawKeyPoints(original, gray, kp):
    outImg = original.copy()
    cv2.drawKeypoints(gray, kp, outImg, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return outImg


sift_im = drawKeyPoints(im.original, im.gray, sift_kp)
surf_im = drawKeyPoints(im.original, im.gray, surf_kp)
kaze_im = drawKeyPoints(im.original, im.gray, kaze_kp)

detectors = 3
cols = detectors + 2
ax = figure.add_subplot(nimages, cols, (i * cols) + 1)
ax.set_axis_off()
ax.imshow(resized_im_original, interpolation='nearest')

ax = figure.add_subplot(nimages, cols, (i * cols) + 2)
ax.set_axis_off()
ax.imshow(resized_im_gray, interpolation='nearest')

ax = figure.add_subplot(nimages, cols, (i * cols) + 3)
ax.set_axis_off()
ax.imshow(sift_im, interpolation='nearest')

ax = figure.add_subplot(nimages, cols, (i * cols) + 4)
ax.set_axis_off()
ax.imshow(surf_im, interpolation='nearest')

ax = figure.add_subplot(nimages, cols, (i * cols) + 5)
ax.set_axis_off()
ax.imshow(kaze_im, interpolation='nearest')
i += 1

canvas.print_figure('feature_extraction_compare.png')
