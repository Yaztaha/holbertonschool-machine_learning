#!/usr/bin/env python3
""" Viterbi algo """
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """ Viterbi algo function """
    T = Observation.shape[0]
    N, M = Emission.shape

    if ((len(Observation.shape)) != 1) or (
            not isinstance(Observation, np.ndarray)):
        return None, None
    if ((len(Emission.shape)) != 2) or (not isinstance(Emission, np.ndarray)):
        return None, None
    N1_T, N2_T = Transition.shape
    if ((len(Transition.shape)) != 2) or (N != N1_T) or (N != N2_T):
        return None, None
    if (N1_T != N2_T) or (not isinstance(Transition, np.ndarray)):
        return None, None
    probability = np.ones((1, N1_T))
    if not (np.isclose((np.sum(Transition, axis=1)), probability)).all():
        return None, None
    if ((len(Initial.shape)) != 2) or (not isinstance(Initial, np.ndarray)):
        return None, None
    if (N != Initial.shape[0]):
        return None, None

    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T))

    # $ \pi_q\beta_,x_1 $
    viterbi[:, 0] = Initial.T * Emission[:, Observation[0]]
    backpointer[:, 0] = 0

    # $\\sum_{q'}^{}\alpha_{q',t-1}A_{q',q}B_{q,x_t}\$
    for t in range(1, T):
        for s in range(N):
            first_part = viterbi[:, t - 1] * Transition[:, s]
            second_part = Emission[s, Observation[t]]
            viterbi[s, t] = np.max(first_part * second_part)
            backpointer[s, t] = np.argmax(first_part * second_part)

    bestpathprob = [0 for i in range(T)]
    bestpathprob[-1] = np.argmax(viterbi[:, T - 1])
    for t in range(T - 1, 0, -1):
        bestpathprob[t - 1] = int(backpointer[bestpathprob[t], t])

    P = np.max(viterbi[:, -1])

    return bestpathprob, P
