import os.path
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from skimage.transform import resize

from src.rougier import voronoi

PALETTE = (
    "#ffea00",
    "#d80003",
    "#fc0093",
    "#cd014a",
    "#00459d",
    "#0097e3",
    "#89cae6",
    "#0095cf",
    "#ff7101",
    "#017939",
    "#55b829",
    "#28b4b1",
    "#655e32",
    "#5b127b",
    "#b50581",
    "#ba5b00",
    "#68250b",
    "#45454d",
    "#b2b3b8",
    "#000000",
    "#ffffff"
)


def _check_size(img, max_dim=1000):
    h, w = img.shape[0], img.shape[1]
    aspect_ratio = h / w
    if (h > w) & (h > max_dim):  # Too tall
        new_h = max_dim
        new_w = int(new_h / aspect_ratio)
        img = resize(img, (new_h, new_w))
    elif (h < w) & (w > max_dim):  # Too wide
        new_w = max_dim
        new_h = int(new_w * aspect_ratio)
        img = resize(img, (new_h, new_w))
    return img


def normalize(D):
    Vmin, Vmax = D.min(), D.max()
    if Vmax - Vmin > 1e-5:
        D = (D - Vmin) / (Vmax - Vmin)
    else:
        D = np.zeros_like(D)
    return D


def initialization(n, D):
    """
    Return n points distributed over [xmin, xmax] x [ymin, ymax]
    according to (normalized) density distribution.

    with xmin, xmax = 0, density.shape[1]
         ymin, ymax = 0, density.shape[0]

    The algorithm here is a simple rejection sampling.
    """

    samples = []
    while len(samples) < n:
        # X = np.random.randint(0, D.shape[1], 10*n)
        # Y = np.random.randint(0, D.shape[0], 10*n)
        X = np.random.uniform(0, D.shape[1], 10 * n)
        Y = np.random.uniform(0, D.shape[0], 10 * n)
        P = np.random.uniform(0, 1, 10 * n)
        index = 0
        while index < len(X) and len(samples) < n:
            x, y = X[index], Y[index]
            x_, y_ = int(np.floor(x)), int(np.floor(y))
            if P[index] < D[y_, x_]:
                samples.append([x, y])
            index += 1
    return np.array(samples)


def stipple(n_point, n_iter, density, force, dat_filename):
    density = density[::-1, :]
    density_P = density.cumsum(axis=1)
    density_Q = density_P.cumsum(axis=1)

    # Initialization
    if not os.path.exists(dat_filename) or force:
        points = initialization(n_point, density)
        print("Nb points:", n_point)
        print("Nb iterations:", n_iter)
    else:
        points = np.load(dat_filename)
        print("Nb points:", len(points))
        print("Nb iterations: -")

    xmin, xmax = 0, density.shape[1]
    ymin, ymax = 0, density.shape[0]
    bbox = np.array([xmin, xmax, ymin, ymax])

    if not os.path.exists(dat_filename) or force:
        for _ in tqdm.trange(n_iter):
            regions, points = voronoi.centroids(points, density, density_P, density_Q)

    return density, points, bbox
