#!/usr/bin/env python3
""" TensorFlow forward propagation """
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_size=[], activations=[]):
    """ forward propagation function """
    for i in range(len(layer_size)):
        layer = create_layer(x, layer_size[i], activations[i])
    return layer
