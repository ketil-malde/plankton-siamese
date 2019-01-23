from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, Model, Input, load_model
from keras.layers import Dense, Activation, Flatten, GlobalAveragePooling2D, Concatenate
from keras.optimizers import SGD
from keras import backend as K

# Load Inception minus the final prediction layer

in_dim = (299,299,3)
# out_dim = 128

tf_device = '/cpu:0'

def create_base_network(input_dim):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_dim)
    # Use 299x299x1 since monochromatic?

    # Load weights from plankton-learn

    # Add a new 128-bit output vector
    tmp = GlobalAveragePooling2D()(base_model.output)
    bitvector = Dense(128, activation='sigmoid')(tmp)
    return Model(inputs=base_model.input, outputs=bitvector)


def create_trivial():
    base_model = Sequential()
    # base_model.add(Flatten(input_shape=in_dim))
    base_model.add(Dense(256, input_dim = 1024))
    base_model.add(Activation('relu'))
    base_model.add(Dense(128))
    base_model.add(Activation('sigmoid'))
    return base_model

# old_model = load_model('models/pretrained.model')
## cut last layer
# bitvector = Dense(128, activation='sigmoid')(tmp)
# base_model = Model(inputs=old_model.input, outputs=bitvector)

base_model = create_base_network(in_dim)

# base_model.summary()

anc_in = Input(shape=in_dim)
pos_in = Input(shape=in_dim)
neg_in = Input(shape=in_dim)

anc_out = base_model(anc_in)
pos_out = base_model(pos_in)
neg_out = base_model(neg_in)

out_vector = Concatenate()([anc_out, pos_out, neg_out])

model = Model(inputs=[anc_in, pos_in, neg_in], outputs=out_vector)

# model.summary()

# Basic triplet loss?
# Note, this learns nothing when dneg>dpos+alpha
def std_triplet_loss(y_true, y_pred, alpha=5):
    # split the prediction vector

    anchor = y_pred[:,0:128]
    pos = y_pred[:,128:256]
    neg = y_pred[:,256:384]

    pos_dist = K.sum(K.square(anchor-pos),axis=1)
    neg_dist = K.sum(K.square(anchor-neg),axis=1)

    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss,0.0)
 
    return loss

def geom_triplet_loss(y_true, y_pred, alpha=5):

    anchor = y_pred[:,0:128]
    pos = y_pred[:,128:256]
    neg = y_pred[:,256:384]

    pos_dist = K.sum(K.square(anchor-pos),axis=1)
    neg_dist = K.sum(K.square(anchor-neg),axis=1)

    basic_loss = pos_dist + alpha/neg_dist
    loss = K.maximum(basic_loss,0.0)  # should never happen
 
    return loss

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss=std_triplet_loss)

# we use SGD with a low learning rate
# model.save('models/initial.model')
