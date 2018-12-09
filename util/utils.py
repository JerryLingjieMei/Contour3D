import os

DATA_FOLDER = "data"
CONTOUR_DATA_FOLDER = os.path.join(DATA_FOLDER, "contour")
CONTOUR_CONTOUR_FOLDER = os.path.join(CONTOUR_DATA_FOLDER, "contours")
CONTOUR_HEIGHTMAP_FOLDER = os.path.join(CONTOUR_DATA_FOLDER, "heightmaps")
CONTOUR_DEPTHMAP_FOLDER = os.path.join(CONTOUR_DATA_FOLDER, "depthmaps")

FULL_DATA_FOLDER = os.path.join(DATA_FOLDER, "full")
FULL_CONTOUR_FOLDER = os.path.join(FULL_DATA_FOLDER, "contours")
FULL_HEIGHTMAP_FOLDER = os.path.join(FULL_DATA_FOLDER, "heightmaps")
FULL_DEPTHMAP_FOLDER = os.path.join(FULL_DATA_FOLDER, "depthmaps")


if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)

if not os.path.exists(CONTOUR_DATA_FOLDER):
    os.mkdir(CONTOUR_DATA_FOLDER)

if not os.path.exists(CONTOUR_CONTOUR_FOLDER):
    os.mkdir(CONTOUR_CONTOUR_FOLDER)

if not os.path.exists(CONTOUR_HEIGHTMAP_FOLDER):
    os.mkdir(CONTOUR_HEIGHTMAP_FOLDER)

if not os.path.exists(CONTOUR_DEPTHMAP_FOLDER):
    os.mkdir(CONTOUR_DEPTHMAP_FOLDER)

if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)


if not os.path.exists(FULL_DATA_FOLDER):
    os.mkdir(FULL_DATA_FOLDER)

if not os.path.exists(FULL_CONTOUR_FOLDER):
    os.mkdir(FULL_CONTOUR_FOLDER)

if not os.path.exists(FULL_HEIGHTMAP_FOLDER):
    os.mkdir(FULL_HEIGHTMAP_FOLDER)

if not os.path.exists(FULL_DEPTHMAP_FOLDER):
    os.mkdir(FULL_DEPTHMAP_FOLDER)
