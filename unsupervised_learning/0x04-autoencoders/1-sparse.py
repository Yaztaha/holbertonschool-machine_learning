#!/usr/bin/env python3
""" Sparse autoencoder """
import tensorflow.keras as keras


def sparse(input_dims, hidden_layers, latent_dims, lambtha):
    """ sparse autoencoder function """
    input_encoder = keras.Input(shape=(input_dims, ))
    input_decoder = keras.Input(shape=(latent_dims, ))
    sparsity = keras.regularizers.l1(lambtha)

    encoded = keras.layers.Dense(hidden_layers[0],
                                 activity_regularizer=sparsity,
                                 activation='relu')(input_encoder)
    for i in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(hidden_layers[i],
                                     activity_regularizer=sparsity,
                                     activation='relu')(encoded)

    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    encoder = keras.Model(inputs=input_encoder, outputs=latent)

    decoded = keras.layers.Dense(hidden_layers[-1],
                                 activation='relu')(input_decoder)
    for j in range(len(hidden_layers) - 2, -1, -1):
        decoded = keras.layers.Dense(hidden_layers[j],
                                     activation='relu')(decoded)
    last = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(inputs=input_decoder, outputs=last)

    encoder_output = encoder(input_encoder)
    decoder_output = decoder(encoder_output)
    auto = keras.Model(inputs=input_encoder, outputs=decoder_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
