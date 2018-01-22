import numpy as np
from matplotlib import pyplot as plt


def average_length(points):
    return np.average(np.linalg.norm(points, axis=1))


def center_mass(points):
    return (np.sum(points, axis=0) / len(points))


def q3_b_compute_transform(points):
    tx, ty = - center_mass(points)
    centered_points = np.add(points, [tx, ty])
    avg_length = average_length(centered_points)
    s = np.sqrt(3) / (2 * avg_length)
    return np.asarray([[s, 0, s * tx], [0, s, s * ty], [0, 0, 1]])


def q3_c_transform_and_plot():
    n_points = 50
    points = np.random.rand(n_points, 2) * 100 + 100
    T = q3_b_compute_transform(points)
    homogeneous_points = np.ones((50, 3))
    homogeneous_points[:, :-1] = points
    transformed_homogeneous_points = np.matmul(T, np.transpose(homogeneous_points))
    transformed_points = np.transpose(transformed_homogeneous_points[:-1, :])

    points_center = center_mass(points)
    transformed_points_center = center_mass(transformed_points)
    plt.figure(figsize=(14, 7), dpi=72)

    plt.subplot(121), plt.scatter(points[:, 0], points[:, 1]), plt.scatter(points_center[0], points_center[1],
                                                                           color='red')
    plt.title('Original Points')  # , plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.scatter(transformed_points[:, 0], transformed_points[:, 1]), plt.scatter(
        transformed_points_center[0], transformed_points_center[1], color='red')
    plt.title('Transformed Points')  # , plt.xticks([]), plt.yticks([])

    plt.tight_layout()
    plt.show()


q3_c_transform_and_plot()
