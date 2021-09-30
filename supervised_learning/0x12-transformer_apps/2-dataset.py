#!/usr/bin/env python3
""" Dataset module """
import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset():
    """ Dataset class"""
    def __init__(self):
        """ class constructor """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.\
            tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """ tockenize dataset method """
        token_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        token_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        return token_pt, token_en

    def encode(self, pt, en):
        """ encode method """
        en_tockens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]
        pt_tockens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]

        return pt_tockens, en_tokens

      def tf_encode(self, pt, en):
        """ tf_encode method """
        pt_t, en_t = tf.py_function(self.encode,
                                              [pt, en],
                                              [tf.int64, tf.int64])
        pt_t.set_shape([None])
        en_t.set_shape([None])

        return pt_t, en_t
