#!/usr/bin/env python3
""" Sklearn GMM """
import sklearn.mixture


def gmm(X, k):
    """ GMM function """
    GMM = sklearn.mixture.GaussianMixture(n_components=k)
    params = GMM.fit(X)
    pi = params.weights_
    m = params.means_
    S = params.covariances_
    clss = GMM.predict(X)
    bic = GMM.bic(X)
    return pi, m, S, clss, bic
