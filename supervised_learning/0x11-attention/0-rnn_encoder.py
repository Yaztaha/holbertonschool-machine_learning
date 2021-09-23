#!/usr/bin/env python3
""" RNN encoder module """
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """ class init """

    def __init__(self, vocab, embedding, units, batch):
        """ init method """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )
        self.gru = tf.keras.layers.GRU(
            units=units,
            recurrent_initializer='glorot_uniform',
            return_sequences=True,
            return_state=True
        )

    def call(self, x, initial):
        """ call method """
        embeddings = self.embedding(x)
        outputs, hidden = self.gru(
            embeddings,
            initial_state=initial
        )
        return outputs, hidden

    def initialize_hidden_state(self):
        """ init hs method """
        init_t = tf.keras.initializers.Zeros()
        return init_t(
            shape=(self.batch, self.units)
        )
