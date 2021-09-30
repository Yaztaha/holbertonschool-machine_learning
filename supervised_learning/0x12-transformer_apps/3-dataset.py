#!/usr/bin/env python3
""" Dataset module """
import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset():
    """ Dataset class"""
    def __init__(self):
        """ class constructor """
        def filter_max_len(x, y, max_length=max_len):
            return tf.logical_and(tf.size(x) <= max_length,
                                  tf.size(y) <= max_length)

        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)

        self.data_train = examples['train']
        self.data_valid = examples['validation']

        PT, EN = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt, self.tokenizer_en = PT, EN

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_train = self.data_train.filter(filter_max_length)
        self.data_train = self.data_train.cache()

        shu = metadata.splits['train'].num_examples
        self.data_train = self.data_train.shuffle(shu)
        pad_shape = ([None], [None])
        self.data_train = self.data_train.padded_batch(batch_size,
                                                       padded_shapes=pad_shape)
        aux = tf.data.experimental.AUTOTUNE
        self.data_train = self.data_train.prefetch(aux)

        self.data_valid = self.data_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(filter_max_length)
        self.data_valid = self.data_valid.padded_batch(batch_size,
                                                       padded_shapes=pad_shape)

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
