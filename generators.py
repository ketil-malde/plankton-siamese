import os
import random
import config as C

def mk_triplet(directory):
    classes = os.listdir(directory)
    images = [os.listdir(os.path.join(directory,x)) for x in classes]

    # pick random positive class
    pos_class = random.randint(0,len(classes)-1)
    # print('Anchor: ',pos_class,classes[pos_class])

    # pick random, different negative class
    neg_class = random.randint(0,len(classes)-2)
    if neg_class >= pos_class:
        neg_class = neg_class + 1
    # print('Negative: ',neg_class,classes[neg_class])

    # pick two random images from class
    anchor = os.path.join(directory, classes[pos_class], random.choice(images[pos_class]))
    pos    = os.path.join(directory, classes[pos_class], random.choice(images[pos_class]))
    neg    = os.path.join(directory, classes[neg_class], random.choice(images[neg_class]))

    # print('Selection:',anchor,pos,neg)
    return(pos_class,neg_class,anchor,pos,neg)

from PIL import Image
import numpy as np

# Scale to image size, paste on white background
def paste(img):
    i = np.ones((299,299,3))
    # NB: Mono images lack the third dimension and will fail here:
    (x,y,z) = img.shape
    start_x = int((299-x)/2)
    end_x   = start_x + x
    start_y = int((299-y)/2)
    end_y   = start_y + y
    i[start_x:end_x,start_y:end_y,:] = img
    return i

def triplet_generator(batch_size,cache_size,directory):
    while True:
        ys = []
        ans = []
        pss = []
        ngs = []
        for i in range(0,batch_size):
            pc,nc,anc,pos,neg = mk_triplet(directory)
            ys.append((pc,nc))
            a_img = np.array(Image.open(anc))/256
            p_img = np.array(Image.open(pos))/256
            n_img = np.array(Image.open(neg))/256
            # Todo: paste it into the middle of a img_size'd canvas
            ans.append(paste(a_img))
            pss.append(paste(p_img))
            ngs.append(paste(n_img))
            # todo: augmentation

        a = np.asarray(ans)
        p = np.asarray(pss)
        n = np.asarray(ngs)
        y = np.asarray(ys)

        yield [a,p,n], y

# Testing:    
g = triplet_generator(4, None, C.train_dir)
for x in range(0,4):
    [a,p,n], y = next(g)
    print(x, "a:", a.shape, "p:", p.shape, "n:", n.shape, "y:", y.shape)
    
