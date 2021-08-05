#!/usr/bin/env python3
""" K-means using sklearn """
import sklearn.cluster


def kmeans(X, k):
    """ kmeans function """
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_
    return C, clss
