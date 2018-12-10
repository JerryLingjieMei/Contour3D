import math

import numpy as np

from dataset.utils import *

class Surface:
    def __init__(self, shapes, size):
        self.size = size
        self.basics = shapes

        self.x, self.y = np.meshgrid(np.arange(0, size[0]), np.arange(0, size[1]))
        self.z = np.zeros(size)
        for f in self.basics:
            self.z += f(self.x, self.y)
        self.lowest = 1e9
        for x in range(SURFACE_SIZE):
            for y in range(SURFACE_SIZE):
                h = 0
                for f in self.basics:
                    h += f(x, y)
                self.lowest = min(self.lowest, h)

    def get_height(self, x, y):
        h = -self.lowest
        ratio = margin_cut_function(x)*margin_cut_function(y)
        for f in self.basics:
            h += f(x, y)
        h *= ratio
        #return np.zeros(h.shape)
        return h

    def normal_vector(self, x, y, z):
        shape = x.shape
        return np.cross(np.stack((np.full(shape, DELTA), np.zeros(shape), self.get_height(x + DELTA, y) - z), axis=-1),
                        np.stack((np.zeros(shape), np.full(shape, DELTA), self.get_height(x, y + DELTA) - z), axis=-1))

    def get_surface(self):
        return self.z

    def get_contours(self, n_contours=10, n_samples=1000, sight_angle=45):
        perspective = normalize(np.array([1, 1, -math.tan(sight_angle / 180 * math.pi)]))
        px = normalize(np.cross(perspective, np.array([0, 0, 1])))
        py = normalize(np.cross(px, perspective))
        
        xl = np.inner(np.array([0, SURFACE_SIZE, 0]), px)
        xr = np.inner(np.array([SURFACE_SIZE, 0, 0]), px)
        yl = np.inner(np.array([0, 0, 0]), py)
        yr = np.inner(np.array([SURFACE_SIZE, SURFACE_SIZE, 0]), py)
        
        res = []
        now = []
        
        depsum = np.zeros([DEPTHMAP_SIZE, DEPTHMAP_SIZE])
        depcnt = np.zeros([DEPTHMAP_SIZE, DEPTHMAP_SIZE])
        
        buckets = [-1e9] * (BUCKET_SIZE*2+1 + 1)
        
        half_contours = (n_contours+1)//2
        for k in range(-half_contours, n_contours + half_contours + 1, 1):
            last_x = SURFACE_SIZE / (n_contours - 1) * (k - 1)
            current_x = SURFACE_SIZE / (n_contours - 1) * k
            if k > 0 and k < n_contours:
                gap_l = max(math.ceil(last_x / SURFACE_SIZE * BUCKET_SIZE),0)
                gap_r = min(math.ceil(current_x / SURFACE_SIZE * BUCKET_SIZE)-1,BUCKET_SIZE)
                #print(gap_l/ BUCKET_SIZE * SURFACE_SIZE,gap_r/ BUCKET_SIZE * SURFACE_SIZE)
                for i in range(gap_l, gap_r + 1):
                    xi = np.full((BUCKET_SIZE+1,), i)
                    yi = np.arange(0, BUCKET_SIZE+1, 1)
                    xs = xi / BUCKET_SIZE * SURFACE_SIZE
                    ys = yi / BUCKET_SIZE * SURFACE_SIZE
                    zs = self.get_height(xs, ys)
                    bids = xi - yi + BUCKET_SIZE
                    for x, y, z, bid in zip(xs, ys, zs, bids):
                        p = np.array([x, y, z])
                        gx = float(np.inner(p, px))
                        gy = float(np.inner(p, py))
                        dep = float(np.inner(p, perspective))
                        if bid < 0 or bid > BUCKET_SIZE*2:
                            raise ValueError
                        if gy > buckets[bid]:
                            buckets[bid] = gy
                            depx = to_index_range((gx-xl)/(xr-xl), DEPTHMAP_SIZE)
                            depy = to_index_range((gy-yl)/(yr-yl), DEPTHMAP_SIZE)
                            depsum[depx][depy] += dep
                            depcnt[depx][depy] += 1
            #print(current_x)
            xs = np.full((n_samples,), current_x)
            ys = np.arange(0, n_samples, 1) * (SURFACE_SIZE * 2 / (n_samples - 1)) - SURFACE_SIZE / 2
            zs = self.get_height(xs, ys)
            nvs = self.normal_vector(xs, ys, zs)
            bids = np.ceil((xs - ys) / SURFACE_SIZE * BUCKET_SIZE).astype(int) + BUCKET_SIZE
            for x, y, z, nv, bid in zip(xs, ys, zs, nvs, bids):
                if x+y<0 or  x+y>SURFACE_SIZE*2 or x-y>SURFACE_SIZE or x-y<-SURFACE_SIZE:
                    continue
                p = np.array([x, y, z])
                gx = float(np.inner(p, px))
                gy = float(np.inner(p, py))
                dep = float(np.inner(p, perspective))
                if np.inner(perspective, nv) < 0 and (bid<0 or bid>BUCKET_SIZE*2 or gy > buckets[bid]):
                    now.append((gx, gy))
                    if bid>=0 and bid<=BUCKET_SIZE*2:
                        buckets[bid] = float(gy)
                        depx = to_index_range((gx-xl)/(xr-xl), DEPTHMAP_SIZE)
                        depy = to_index_range((gy-yl)/(yr-yl), DEPTHMAP_SIZE)
                        depsum[depx][depy] += dep
                        depcnt[depx][depy] += 1
                elif len(now) != 0:
                    res.append(now)
                    now = []
            if len(now) != 0:
                res.append(now)
                now = []
        
        depl = np.inner(np.array([0, 0, 0]), perspective)
        depr = np.inner(np.array([SURFACE_SIZE, SURFACE_SIZE, 0]), perspective)
        
        for x in range(DEPTHMAP_SIZE):
            head = 0
            tail = DEPTHMAP_SIZE-1
            while head <= tail and depcnt[x][head] == 0:
                depsum[x][head] = (head+.5)/DEPTHMAP_SIZE*(depr-depl)+depl
                depcnt[x][head] = 1
                head+=1
            while head <= tail and depcnt[x][tail] == 0:
                depsum[x][tail] = (tail+.5)/DEPTHMAP_SIZE*(depr-depl)+depl
                depcnt[x][tail] = 1
                tail-=1
            if head>tail:
                print(x)
            i = head
            while i <= tail:
                j = i
                while j<=tail and depcnt[x][j] == 0:
                    j += 1
                if i < j:
                    a = depsum[x][i-1]/depcnt[x][i-1]
                    b = depsum[x][j]/depcnt[x][j]
                    for k in range(i,j):
                        depsum[x][k]=((k-i+1)*b+(j-k)*a)/(j-i+1)
                        depcnt[x][k]=1
                    i = j
                else:
                    i = j + 1
        depmap = depsum/depcnt
        return res, depmap, xl, xr, yl, yr
