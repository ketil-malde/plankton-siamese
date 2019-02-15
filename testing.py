import config as C

from PIL import Image
import numpy as np
from generators import paste

def class_file(model, fname):
    img = np.array(Image.open(fname))/256
    return model.predict(np.expand_dims(paste(img), axis=0))

# Calculate histogram of all distances from v in v1 to w in v2
def dist_hist(v1,v2):
    return [np.linalg.norm(v-w) for v in v1 for w in v2]

def centroid(vs):
    x0 = np.zeros_like(vs[0])
    for x in vs:
        x0 = np.add(x0,x)
    return x0/len(vs)

import os

def run_test(model, tdir=C.test_dir):
    # load a bunch of images, calculate outputs
    classes = os.listdir(tdir)
    res = {}
    for c in classes:
           images = os.listdir(os.path.join(tdir,c))
           vectors = [class_file(model, os.path.join(tdir,c,f)) for f in images]

           # pick the centroid image
           # pick the worst cases?
           ds = dist_hist(vectors,vectors)
           print('%s average radius: %.4f worst case distance: %.4f' % (c, sum(ds)/len(ds), max(ds)))

           print('Distances:',end='')
           cent = centroid(vectors)
           for n in res:
               print('%s: %.3f' % (n, np.linalg.norm(cent-res[n])), end=' ')
           res[c]=cent
           print()
