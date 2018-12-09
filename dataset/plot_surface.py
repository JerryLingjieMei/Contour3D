from dataset.utils import *
from dataset.make_surface import generate_surface
from lib.utils import *
import matplotlib.pyplot as plt
import os
import argparse
from multiprocessing import Pool, cpu_count


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--start_index', help='image index to start', type=int)
    parser.add_argument('--end_index', help='image index to end', type=int)
    parser.add_argument("--stride", help="image index stride", type=int, default=1)
    return parser.parse_args()


def main(case_id):
    np.random.seed()
    surface = generate_surface()
    height_map = surface.get_surface()
    np.save(os.path.join(HEIGHTMAP_FOLDER, "{:05d}.npy".format(case_id)), np.array(height_map))
    contours, depmap, xl, xr, yl, yr = surface.get_contours(n_contours=20, n_samples=500, sight_angle=53)
    np.save(os.path.join(DEPTHMAP_FOLDER, "{:05d}.npy".format(case_id)), np.array(depmap))
    #print(depmap)
    plt.clf()
    for contour in contours:
        contour = [p for p in contour if p[0]>=xl and p[0]<=xr and p[1]>=yl and p[1]<=yr]
        x = np.array([p[0] for p in contour])
        y = np.array([p[1] for p in contour])
        plt.plot(x, y, color='black')
    plt.xlim(xl, xr)
    plt.ylim(yl, yr)
    plt.axis('off')
    plt.savefig(os.path.join(CONTOUR_FOLDER, "{:05d}.png".format(case_id)), bbox_inches='tight', pad_inches=0)
    print("{:05d} generated".format(case_id))


if __name__ == '__main__':
    args = parse_args()
    worker_args = []
    for i in range(args.start_index, args.end_index, args.stride):
        worker_args.append((i,))

    with Pool(cpu_count()) as p:
        p.starmap(main, worker_args)
    