# The imported generators expect to find training data in data/train
# and validation data in data/validation
from generator import train_generator, validation_generator

from keras.models import load_model
from keras.callbacks import CSVLogger
from keras.optimizers import SGD

# Modify this to match last version saved
last = 0


def save_name(i):
    return ('models/epoch_'+str(i)+'.model')


if last == 0:
    model = load_model('models/initial.model')
else:
    model = load_model(save_name(last))

# Use log to file
logger = CSVLogger('train.log', append=True, separator='       ')


def train_step(i):
    model.fit_generator(
        train_generator, steps_per_epoch=1000, epochs=10,
        callbacks=[logger],
        validation_data=validation_generator, validation_steps=500)
    model.save(save_name(i))


# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['mse', 'accuracy'])

for i in range(last+1, last+10):
        print('Starting iteration '+str(i))
        train_step(i)
