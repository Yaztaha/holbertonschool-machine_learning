#!/usr/bin/env python3
""" TF-IDF embedding """

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """ TF-IDF embedding function """

    tfidf = TfidfVectorizer(vocabulary=vocab)
    output = tfidf.fit_transform(sentences)
    tfidf.get_feature_names()
    return output.toarray(), tfidf.get_feature_names()
