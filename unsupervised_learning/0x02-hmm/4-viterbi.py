#!/usr/bin/env python3
""" """
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """ """
    try:
        T = Observation.shape[0]
        N = Transition.shape[0]
        omega = np.zeros((T, N))
        omega[0, :] = np.log(Initial.T * Emission[:, Observation[0]])
        prev = np.zeros((T - 1, N))
        for t in range(1, T):
            for j in range(N):
                a1 = omega[t-1]
                a2 = np.log(Transition[:, j])
                a3 = np.log(Emission[j, Observation[t]])
                probability = (omega[t - 1] + np.log(Transition[:, j]) +
                               np.log(Emission[j, Observation[t]]))
                prev[t - 1, j] = np.argmax(probability)
                omega[t, j] = np.max(probability)
        S = np.zeros(T)
        last_state = np.argmax(omega[T - 1, :])
        S[0] = last_state
        backtrack_index = 1
        for i in range(T - 2, -1, -1):
            S[backtrack_index] = prev[i, int(last_state)]
            last_state = prev[i, int(last_state)]
            backtrack_index += 1
        S = np.flip(S, axis=0)
        result = []
        for s in S:
            result.append(int(s))
        P = np.max(np.exp(omega[-1:, :]))
        return (P, result)
    except Exception:
        return None, None
