# The imported generators expect to find training data in data/train
# and validation data in data/validation
from keras.models import load_model
from keras.callbacks import CSVLogger
from keras.optimizers import SGD

from create_model import create_base_network, in_dim, tripletize, std_triplet_loss
from generators import triplet_generator
from testing import run_test

import config as C

last = 0

def save_name(i):
    return ('models/epoch_'+str(i)+'.model')

# Use log to file
logger = CSVLogger('train.log', append=True, separator='\t')

def train_step():
    model.fit_generator(
        triplet_generator(C.batch_size, None, C.train_dir), steps_per_epoch=1000, epochs=10,
        callbacks=[logger],
        validation_data=triplet_generator(C.batch_size, None, C.val_dir), validation_steps=100)

base_model = create_base_network(in_dim)
model = tripletize(base_model)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss=std_triplet_loss)

for i in range(last+1, last+10):
        print('Starting iteration '+str(i))
        train_step()
        base_model.save(save_name(i))
        with open('train.log', 'a') as f:
            run_test(base_model, outfile=f)
