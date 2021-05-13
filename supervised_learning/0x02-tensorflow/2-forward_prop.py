#!/usr/bin/env python3
""" TensorFlow forward propagation """
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_size=[], activations=[]):
    """ forward propagation function """
    ph = create_layer(x, layer_size[0], activations[0])
    for i in range(1, len(layer_size)):
        layer = create_layer(ph, layer_size[i], activations[i])
    return layer
