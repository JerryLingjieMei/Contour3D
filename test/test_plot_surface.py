from dataset import plot_surface
import numpy as np
import matplotlib.pyplot as plt
from dataset.make_surface import generate_surface
from util.utils import *
from dataset.utils import DEFAULT_SIGHT_ANGLE


def main(case_id):
    np.random.seed()
    surface = generate_surface()
    height_map = surface.get_surface()
    np.save(os.path.join("test/output/heightmaps", "{:05d}.npy".format(case_id)), np.array(height_map))
    contours, depmap, xl, xr, yl, yr = surface.get_contours(n_contours=20, n_samples=1000,
                                                            sight_angle=DEFAULT_SIGHT_ANGLE)
    np.save(os.path.join("test/output/depthmaps", "{:05d}.npy".format(case_id)), np.array(depmap))
    plt.figure()
    plt.matshow(depmap, cmap='gray')
    plt.savefig(os.path.join("test/output/depthmaps", "{:05d}.png".format(case_id)))
    plt.clf()
    fig = plt.figure()
    DPI = fig.get_dpi()
    fig.set_size_inches(512 / float(DPI), 512 / float(DPI))
    for contour in contours:
        contour = [p for p in contour if p[0] >= xl and p[0] <= xr and p[1] >= yl and p[1] <= yr]
        x = np.array([p[0] for p in contour])
        y = np.array([p[1] for p in contour])
        plt.plot(x, y, color='black')
    plt.xlim(xl, xr)
    plt.ylim(yl, yr)
    plt.axis('off')
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    fig.subplots_adjust(bottom=0)
    fig.subplots_adjust(top=1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(left=0)
    plt.savefig(os.path.join("test/output/contours", "{:05d}.png".format(case_id)), bbox_inches='tight', pad_inches=0)
    print("{:05d} generated".format(case_id))


if __name__ == '__main__':
    main(0)
