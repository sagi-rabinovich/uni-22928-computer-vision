import cv2

from cifa_challenge.image_dataset import ImageDataset
from cifa_challenge.image_grid_plot import plot_image_grid

image_dataset = ImageDataset()

LABEL_COUNT = len(image_dataset.CIFAR_10_LABELS)
DATA_BATCH_1 = 'data_batch_1'
image_contexts = image_dataset.load_training_data(batches=DATA_BATCH_1)[:20]

image_grid = []
sigmas = [20, 50, 100, 150]
ds = [4]
for img in image_contexts:
    image_row = [img.original]
    for s in sigmas:
        for d in ds:
            filtered = cv2.bilateralFilter(img.original, d, s, s)
            image_row.append(filtered)
    image_grid.append(image_row)
header = ['original']

for s in sigmas:
    for d in ds:
        header.append('s=' + str(s) + ', d=' + str(d))

plot_image_grid(image_grid, (len(image_contexts), len(sigmas) * len(ds) + 1), 'bluring.png', header=None)
