from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD

# Load Inception minus the final prediction layer
base_model = InceptionV3(weights='imagenet',
                         include_top=False,
                         input_shape=(299, 299, 3))
# Use 299x299x1 since monochromatic?

# Add a new 84-class prediction layer
tmp = GlobalAveragePooling2D()(base_model.output)
predictions = Dense(60, activation='softmax')(tmp)
model = Model(inputs=base_model.input, outputs=predictions)

# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                      loss='categorical_crossentropy',
                                    metrics=['mse', 'accuracy'])

model.save('models/initial.model')
