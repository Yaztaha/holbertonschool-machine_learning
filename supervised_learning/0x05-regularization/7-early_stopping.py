#!/usr/bin/env python3
""" Early stopping regularization """


def early_stopping(cost, opt_cost, threshold, patience, count):
    """ early stopping function """
    if (opt_cost - cost) > threshold:
        count = 0
    else:
        count += 1
    if count == patience:
        return True, count
    return False, count
