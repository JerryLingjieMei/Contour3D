import os

DATA_FOLDER = "data"
CONTOUR_FOLDER = os.path.join(DATA_FOLDER, "contours")
HEIGHTMAP_FOLDER = os.path.join(DATA_FOLDER, "heightmaps")
DEPTHMAP_FOLDER = os.path.join(DATA_FOLDER, "depthmaps")

if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)

if not os.path.exists(CONTOUR_FOLDER):
    os.mkdir(CONTOUR_FOLDER)

if not os.path.exists(HEIGHTMAP_FOLDER):
    os.mkdir(HEIGHTMAP_FOLDER)

if not os.path.exists(DEPTHMAP_FOLDER):
    os.mkdir(DEPTHMAP_FOLDER)
