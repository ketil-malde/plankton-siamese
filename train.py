# The imported generators expect to find training data in data/train
# and validation data in data/validation
from generators import triplet_generator

from keras.models import load_model
from keras.callbacks import CSVLogger
from keras.optimizers import SGD

from create_model import model

from config import batch_size

last = 0

def save_name(i):
    return ('models/epoch_'+str(i)+'.model')

# Use log to file
logger = CSVLogger('train.log', append=True, separator='       ')

def train_step(i):
    model.fit_generator(
        triplet_generator(batch_size), steps_per_epoch=1000, epochs=10,
        callbacks=[logger],
        validation_data=triplet_generator(batch_size), validation_steps=500)
    model.save(save_name(i))


for i in range(last+1, last+10):
        print('Starting iteration '+str(i))
        train_step(i)
