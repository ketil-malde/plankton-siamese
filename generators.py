# from keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         vertical_flip=True)

# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#         'train',
#         target_size=(299, 299),
#         batch_size=32,
#         class_mode='categorical')

# validation_generator = test_datagen.flow_from_directory(
#         'validate',
#         target_size=(299, 299),
#         batch_size=32,
#         class_mode='categorical')

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
    for i in range(0,batch_size-1):
        pc,nc,anc,pos,neg = mk_triplet()
        ys.append((pc,nc))
        ans.append(np.array(Image.open(anc))/256)
        pss.append(np.array(Image.open(pos))/256)
        ngs.append(np.array(Image.open(neg))/256)
    # todo: augmentation

    a = np.asarray(ans)
    p = np.asarray(pss)
    n = np.asarray(ngs)

    yield(ys,a,p,n)

for x in triplet_generator(32):
    #    print(x)
    print()
    
