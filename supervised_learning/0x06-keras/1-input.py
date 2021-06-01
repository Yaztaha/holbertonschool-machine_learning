#!/usr/bin/env python3
""" Build model """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ method to build a model """
    inp = K.Input(shape=(nx,))
    reg = K.regularizers.l2(lambtha)

    y = K.layers.Dense(layers[0], activation=activations[0],
                       kernel_regularizer=reg, name="dense")(inp)
    for i in range(1, len(activations)):
        inp_next = K.layers.Dropout(1 - keep_prob)(y)
        y = K.layers.Dense(layers[i], activation=activations[i],
                           kernel_regularizer=reg,
                           name="dense_" + str(i))(inp_next)
    model = K.Model(inputs=inp, outputs=y)

    return model
