import config as C

from PIL import Image
import numpy as np
from generators import paste

def class_file(model, fname):
    img = np.array(Image.open(fname))/256
    return model.predict(paste(img))

# Calculate histogram of all distances from v in v1 to w in v2
def dist_hist(v1,v2):
    [np.linalg.norm(v-w) for v in v1 for w in v2]

def centroid(vs):
    x0 = np.zeroes_like(vs[0])
    for x in vs:
        x0 = np.add(x0,x)
    return x0/len(vs)

import os

def run_test(model, tdir=C.test_dir):
    # load a bunch of images, calculate outputs
    classes = os.listdir(tdir)
    res = {}
    for c in classes:
           images = os.listdir(os.path.join(tdir,x))
           vectors = [class_file(model, f) for f in images]
           # pick the centroid image
           # pick the worst cases?
           ds = dist_hist(vectors,vectors)
           print(c,'average radius:',sum(ds)/len(ds), 'worst case distance:', max(ds))
           cent = centroid(vectors)
           for n,ct in res:
               print('Distance to', n,':', np.linalg.norm(cent-ct))

