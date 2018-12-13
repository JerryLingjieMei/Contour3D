from visualizer.visualize import plot_depthmap
from imageio import imread
import os
from util.utils import *
from dataset.utils import *
import torchvision.transforms.functional as TF

if __name__ == '__main__':
    data = imread("results/contour_pix2pix/test_latest/images/20005_real_B.png")
    depth_map = np.rot90((data[:, :, 0] / 256), axes=(1, 0)) * 500
    plot_depthmap(depth_map, DEFAULT_SIGHT_ANGLE, "test/output/full_0.png")
    data = imread("results/contour_pix2pix/test_latest/images/20005_fake_B.png")
    depth_map = np.rot90((data[:, :, 0] / 256), axes=(1, 0)) * 500
    plot_depthmap(depth_map, DEFAULT_SIGHT_ANGLE, "test/output/full_1.png")
