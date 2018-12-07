from dataset.make_surface import generate_surface

if __name__ == '__main__':
    surface = generate_surface()
    surface.get_contours(n_contours=20, n_samples=500, sight_angle=53)
