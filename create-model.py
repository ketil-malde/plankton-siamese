from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense

# Load Inception minus the final prediction layer
base_model = InceptionV3(weights='imagenet',
                         include_top=False,
                         input_shape=(299, 299, 3))

# Add a new 84-class prediction layer
predictions = Dense(84, activation='softmax')(base_model.output)
model = Model(inputs=base_model.input, outputs=predictions)

model.save('models/initial.model')
