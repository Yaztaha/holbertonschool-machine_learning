#!/usr/bin/env python3
""" Moving average """


def moving_average(data, beta):
    """ wheighted moving average function """
    mov_avgs = []
    mov_avg = 0
    for i in range(len(data)):
        mov_avg = beta * mov_avg + (1 - beta) * data[i]
        mov_avgs += [mov_avg / (1 - beta ** (i + 1))]
    return mov_avgs
