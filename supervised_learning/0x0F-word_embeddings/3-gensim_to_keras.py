#!/usr/bin/env python3
""" Convert gensim word2vec to keras """


def gensim_to_keras(model):
    """ gensim to keras funcyion """
    return model.wv.get_keras_embedding(train_embeddings=True)
