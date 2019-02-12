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

import os

def run_test(model, tdir=C.test_dir):
    # load a bunch of images, calculate outputs
    classes = os.listdir(tdir)
    for c in classes:
           images = os.listdir(os.path.join(tdir,x))
           vectors = [class_file(f) for f in images]
           # pick the centroid image
           # pick the worst cases?
           ds = dist_hist(vectors,vectors)
           print(c,'average distance:',sum(ds)/len(ds), 'worst case distance:', max(ds))
