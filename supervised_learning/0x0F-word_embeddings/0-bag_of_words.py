#!/usr/bin/env python3
""" Module 1"""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """ bag of words funcion """

    input_vectorizer = CountVectorizer(vocabulary=vocab)
    output_vectorizer = input_vectorizer.fit_transform(sentences)
    return output_vectorizer.toarray(), input_vectorizer.get_feature_names()
