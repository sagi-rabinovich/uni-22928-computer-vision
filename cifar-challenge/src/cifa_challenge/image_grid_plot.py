import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_template import FigureCanvas
from matplotlib.figure import Figure


def plot_image_grid(img_grid, grid_shape, file_name, normalize=False, header=None):
    plt.ioff()
    figure = Figure(figsize=(10, 10))
    canvas = FigureCanvas(figure)
    for y in range(grid_shape[0]):
        for x in range(grid_shape[1]):
            img = img_grid[y][x]
            if normalize:
                min_ = np.min(img)
                max_ = np.max(img)
                img = (img - min_) / (max_ - min_)
            ax = figure.add_subplot(grid_shape[0], grid_shape[1], (y * grid_shape[1]) + x + 1)
            ax.set_axis_off()
            ax.imshow(img, interpolation='nearest')
            if y == 0 and header is not None:
                ax.set_title(header[x])

    canvas.print_figure(os.path.join('../../results/', file_name))
