#!/usr/bin/env python3
""" Agglomorative clustering """
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """ agglomorative clustering function """
    H = scipy.cluster.hierarchy
    links = H.linkage(X, method='ward')
    clss = H.fcluster(links, t=dist, criterion='distance')
    plt.figure()
    H.dendrogram(links, color_threshold=dist)
    plt.show()
    return clss
