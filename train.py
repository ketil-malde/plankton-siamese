# The imported generators expect to find training data in data/train
# and validation data in data/validation
from keras.models import load_model
from keras.callbacks import CSVLogger
from keras.optimizers import SGD

import os

from create_model import create_base_network, in_dim, tripletize, std_triplet_loss
from generators import triplet_generator
import testing as T

import config as C

last = 0

def save_name(i):
    return ('models/epoch_'+str(i)+'.model')

def log(s):
    with open(C.logfile, 'a') as f:
        print(s, file=f)

# Use log to file
logger = CSVLogger(C.logfile, append=True, separator='\t')

def train_step():
    model.fit_generator(
        triplet_generator(C.batch_size, None, C.train_dir), steps_per_epoch=1000, epochs=C.iterations,
        callbacks=[logger],
        validation_data=triplet_generator(C.batch_size, None, C.val_dir), validation_steps=100)

if last==0:
    log('Creating base network from scratch.')
    if not os.path.exists('models'):
        os.makedirs('models')
    base_model = create_base_network(in_dim)
else:
    log('Loading model:'+save_name(last))
    base_model = load_model(save_name(last))

model = tripletize(base_model)
model.compile(optimizer=SGD(lr=C.learn_rate, momentum=0.9),
              loss=std_triplet_loss())

vs = T.get_vectors(base_model, C.val_dir)
cents = [T.centroid(vs[v]) for v in vs]

for i in range(last+1, last+11):
    log('Starting iteration '+str(i)+' lr='+str(C.learn_rate))
    train_step()
    C.learn_rate = C.learn_rate * C.lr_decay
    base_model.save(save_name(i))

    vs = T.get_vectors(base_model, C.val_dir)
    c = T.count_nearest_centroid(vs)
    with open(C.logfile, 'a') as f:
        T.accuracy_counts(c, outfile=f)
    log('Summarizing '+str(i))
    with open('summarize.'+str(i)+'.log', 'w') as sumfile:
        T.summarize(vs, outfile=sumfile)
    with open('clusters.'+str(i)+'.log', 'w') as cfile:
        T.confusion_counts(c, outfile=cfile)
    c_tmp = [T.centroid(vs[v]) for v in vs]
    c_mv = [T.dist(a,b) for a in cents for b in c_tmp]
    log('Centroid moved: '+str(c_mv))
    cents = c_tmp
