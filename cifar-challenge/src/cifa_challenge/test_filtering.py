import cv2

from cifa_challenge.image_dataset import ImageDataset
from cifa_challenge.image_grid_plot import plot_image_grid

image_dataset = ImageDataset()

LABEL_COUNT = len(image_dataset.CIFAR_10_LABELS)
DATA_BATCH_1 = 'data_batch_1'
image_contexts = image_dataset.load_training_data(batches=DATA_BATCH_1)[:20]

image_grid = []
sigma = [50, 100, 150, 200]
d = 4
for img in image_contexts:
    image_row = [img.original]
    for s in sigma:
        filtered = cv2.bilateralFilter(img.original, d, s, s)
        image_row.append(filtered)
    image_grid.append(image_row)

plot_image_grid(image_grid, (len(image_contexts), len(sigma) + 1), 'bluring.png')
