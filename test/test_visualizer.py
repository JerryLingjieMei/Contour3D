from visualizer.visualize import plot_depthmap
from imageio import imread
import os
from util.utils import *
from dataset.utils import *
import torchvision.transforms.functional as TF

if __name__ == '__main__':
    data = imread("results/contour_pix2pix_short/test_latest/images/20005_real_B.png")
    depth_map = np.rot90((data[:, :, 0] / 256), axes=(1, 0)) * 500
    plot_depthmap(depth_map, DEFAULT_SIGHT_ANGLE, 30, "test/output/20005_real_30.png")
    plot_depthmap(depth_map, DEFAULT_SIGHT_ANGLE, 120, "test/output/20005_real_120.png")
    data = imread("results/contour_pix2pix_short/test_latest/images/20005_fake_B.png")
    depth_map = np.rot90((data[:, :, 0] / 256), axes=(1, 0)) * 500
    plot_depthmap(depth_map, DEFAULT_SIGHT_ANGLE, 30, "test/output/20005_fake_30.png")
    plot_depthmap(depth_map, DEFAULT_SIGHT_ANGLE, 120, "test/output/20005_fake_120.png")
