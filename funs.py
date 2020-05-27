import numpy as np
import math


def l2_loss(d1, d2):
    """
    This function calculate the L2 loss of two arrays
    :param d1: The first np array
    :param d2: The second np array
    :return: MSE
    """
    return np.mean((np.array(d1) - np.array(d2))**2)

def feed_forward(dat, weight, loss_fun):
    """
    This is the feed forward step of a neural network
    :param dat: Input data. Each column
    :param weight:
    :param loss_fun:
    :return:
    """