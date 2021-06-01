#!/usr/bin/env python3
""" Train model w/ early stopping """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """ method train model w/ early stopping """
    early = []
    if early_stopping and validation_data:
        early.append(K.callbacks.EarlyStopping(monitor="val_loss",
                                               patience=patience))

    history = network.fit(x=data, y=labels, epochs=epochs,
                          batch_size=batch_size,
                          validation_data=validation_data,
                          shuffle=shuffle, verbose=verbose, callbacks=early)
    return history
