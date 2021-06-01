#!/usr/bin/env python3
""" Train model with learning rate decay """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True,
                shuffle=False):
    """ method to train a model w/ LRD """
    def scheduler(epoch):
        """inverse time decay"""
        a = alpha / (1 + (decay_rate * epoch))
        return a

    callback = []
    if early_stopping and validation_data:
        callback.append(K.callbacks.EarlyStopping(patience=patience))
    if learning_rate_decay and validation_data:
        callback.append(K.callbacks.LearningRateScheduler(scheduler,
                                                          verbose=1))

    history = network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                          shuffle=shuffle, verbose=verbose, callbacks=callback,
                          validation_data=validation_data)
    return history
