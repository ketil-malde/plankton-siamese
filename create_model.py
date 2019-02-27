from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, Model, Input, load_model
from keras.layers import Dense, Activation, Flatten, GlobalAveragePooling2D, Concatenate
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


# This is just for testing - Inception takes forever to set up in Tensorflow
def create_trivial():
    base_model = Sequential()
    base_model.add(Dense(256, input_dim = 1024))
    base_model.add(Activation('relu'))
    base_model.add(Dense(128))
    base_model.add(Activation('sigmoid'))
    return base_model


def tripletize(bmodel):
    anc_in = Input(shape=in_dim)
    pos_in = Input(shape=in_dim)
    neg_in = Input(shape=in_dim)

    anc_out = bmodel(anc_in)
    pos_out = bmodel(pos_in)
    neg_out = bmodel(neg_in)

    out_vector = Concatenate()([anc_out, pos_out, neg_out])
    return Model(inputs=[anc_in, pos_in, neg_in], outputs=out_vector)

# Basic triplet loss.
# Note, due to the K.maximum, this learns nothing when dneg>dpos+alpha
def std_triplet_loss(alpha=5):
    # split the prediction vector
    def myloss(y_true, y_pred):
        anchor = y_pred[:,0:128]
        pos = y_pred[:,128:256]
        neg = y_pred[:,256:384]
        pos_dist = K.sum(K.square(anchor-pos),axis=1)
        neg_dist = K.sum(K.square(anchor-neg),axis=1)
        basic_loss = pos_dist - neg_dist + alpha
        loss = K.maximum(basic_loss,0.0)
        return loss

    return myloss

# in retrospect, this has some problems, namely that the derivative of 1/x
# goes quickly (quadratically) to zero as x increases.
# I.e. the gradient disappears, and we get very slow learning.
def geom_triplet_loss(alpha=5):
    def myloss(y_true, y_pred):
        anchor = y_pred[:,0:128]
        pos = y_pred[:,128:256]
        neg = y_pred[:,256:384]
        pos_dist = K.sum(K.square(anchor-pos),axis=1)
        neg_dist = K.sum(K.square(anchor-neg),axis=1)
        basic_loss = pos_dist + alpha/neg_dist
        loss = K.maximum(basic_loss,0.0)  # should never happen
        return loss

    return myloss

# By placing the maximum on the loss for negative (and not the total)
# we may still learn to pack clusters after they are acceptably separated.
def alt_triplet_loss(alpha=5):
    def myloss(y_true, y_pred):
        anchor = y_pred[:,0:128]
        pos = y_pred[:,128:256]
        neg = y_pred[:,256:384]
        pos_dist = K.sum(K.square(anchor-pos),axis=1)
        neg_dist = K.sum(K.square(anchor-neg),axis=1)
        loss = pos_dist + K.maximum(alpha - neg_dist, 0.0)
        return loss

    return myloss
