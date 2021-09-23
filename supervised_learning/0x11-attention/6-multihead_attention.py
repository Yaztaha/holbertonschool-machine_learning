#!/usr/bin/env python3
""" multi head module """
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """ multihead class"""

    def __init__(self, dm, h):
        """ clas constructor """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = int(dm / h)
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def split_heads(self, x, batch_size):
        """ split head method """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """ call method """
        batch_size = tf.shape(Q)[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)
        q = self.split_heads(Q, batch_size)
        k = self.split_heads(K, batch_size)
        v = self.split_heads(V, batch_size)
        scaled_attention, weights = sdp_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))
        output = self.linear(concat_attention)
        return output, weights
