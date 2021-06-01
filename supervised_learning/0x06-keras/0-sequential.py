#!/usr/bin/env python3
""" Build model """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ mehtod to build a model """
    model = K.Sequential()
    reg = K.regularizers.l2(lambtha)
    model.add(K.layers.Dense(layers[0], input_shape=(nx,),
              activation=activations[0], kernel_regularizer=reg, name="dense"))
    for i in range(1, len(layers)):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(layers[i],
                  activation=activations[i], kernel_regularizer=reg,
                  name="dense_" + str(i)))
    return model
