#!/usr/bin/env python3
""" Vanilla autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ autoencoder function """
    input_enc = keras.Input(shape=(input_dims, ))
    input_dec = keras.Input(shape=(latent_dims, ))

    encoded = keras.layers.Dense(hidden_layers[0],
                                 activation='relu')(input_enc)
    for i in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(encoded)

    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    encoder = keras.Model(inputs=input_enc, outputs=latent)

    decoded = keras.layers.Dense(hidden_layers[-1],
                                 activation='relu')(input_dec)

    for j in range(len(hidden_layers) - 2, - 1, - 1):
        decoded = keras.layers.Dense(hidden_layers[j],
                                     activation='relu')(decoded)
    last = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(inputs=input_dec, outputs=last)

    encoder_output = encoder(input_enc)
    decoder_output = decoder(encoder_output)
    auto = keras.Model(inputs=input_enc, outputs=decoder_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
