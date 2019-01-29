import os
import random
classes = os.listdir('train')
images = [os.listdir(os.path.join('train',x)) for x in classes]

def mk_triplet():
    # pick random positive class
    pos_class = random.randint(0,len(classes)-1)
    # print('Anchor: ',pos_class,classes[pos_class])

    # pick random, different negative class
    neg_class = random.randint(0,len(classes)-2)
    if neg_class >= pos_class:
        neg_class = neg_class + 1
    # print('Negative: ',neg_class,classes[neg_class])

    # pick two random images from class
    anchor = os.path.join('train',classes[pos_class],random.choice(images[pos_class]))
    pos = os.path.join('train',classes[pos_class],random.choice(images[pos_class]))
    neg = os.path.join('train',classes[neg_class],random.choice(images[neg_class]))

    # print('Selection:',anchor,pos,neg)
    return(pos_class,neg_class,anchor,pos,neg)

from PIL import Image
import numpy as np

def triplet_generator(batch_size):
    ys = []
    ans = []
    pss = []
    ngs = []
    for i in range(0,batch_size):
        pc,nc,anc,pos,neg = mk_triplet()
        ys.append((pc,nc))
        ans.append(np.array(Image.open(anc))/256)
        pss.append(np.array(Image.open(pos))/256)
        ngs.append(np.array(Image.open(neg))/256)
    # todo: augmentation

    a = np.asarray(ans)
    p = np.asarray(pss)
    n = np.asarray(ngs)
    y = np.asarray(ys)

    # print("a:", a.shape, "p:", p.shape, "n:", n.shape, "y:", y.shape)
    
    yield [a,p,n], y
