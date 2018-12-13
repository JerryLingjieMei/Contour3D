import numpy as np
from dataset.utils import *
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import argparse
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from util.utils import CONTOUR_CONTOUR_FOLDER, CONTOUR_HEIGHTMAP_FOLDER, CONTOUR_DEPTHMAP_FOLDER


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--depthmap', help='the depthmap ', type=str)
    return parser.parse_args()


def plot_depthmap(depthmap, sight_angle, save_fig):
    perspective = normalize(np.array([1, 1, -math.tan(sight_angle / 180 * math.pi)]))
    px = normalize(np.cross(perspective, np.array([0, 0, 1])))
    py = normalize(np.cross(px, perspective))

    xl = np.inner(np.array([0, SURFACE_SIZE, 0]), px)
    xr = np.inner(np.array([SURFACE_SIZE, 0, 0]), px)
    yl = np.inner(np.array([0, 0, 0]), py)
    yr = np.inner(np.array([SURFACE_SIZE, SURFACE_SIZE, 0]), py)

    X = []
    Y = []
    Z = []

    for i in range(0, DEPTHMAP_SIZE, 5):
        X.append([])
        Y.append([])
        Z.append([])
        for j in range(0, DEPTHMAP_SIZE, 5):
            gx = (i + .5) / DEPTHMAP_SIZE * (xr - xl) + xl
            gy = (j + .5) / DEPTHMAP_SIZE * (yr - yl) + yl
            p = np.array([0, 0, 0]) + px * gx + py * gy + perspective * depthmap[i][j]
            X[-1].append(p[0])
            Y[-1].append(p[1])
            Z[-1].append(p[2])

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    axe_min = np.min(X)
    axe_max = np.max(X)
    z_min = np.min(Z)
    z_max = np.max(Z)
    delta_z = (z_min + z_max) / 2 - (axe_min + axe_max) / 2
    ax.set_xlim(axe_min, axe_max)
    ax.set_ylim(axe_min, axe_max)
    ax.set_zlim(axe_min + delta_z, axe_max + delta_z)
    plt.savefig(save_fig)
