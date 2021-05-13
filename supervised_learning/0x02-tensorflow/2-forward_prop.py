#!/usr/bin/env python3
""" TensorFlow forward propagation """
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_size=[], activations=[]):
    """ forward propagation function """
    lyr = create_layer(x, layer_size[0], activations[0])
    for i in range(1, len(layer_size)):
        lyr = create_layer(lyr, layer_size[i], activations[i])

    return lyr
