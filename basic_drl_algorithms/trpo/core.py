import numpy as np
import torch.nn as nn
import scipy.signal
from gym.spaces import Box, Discrete


def combined_shape(length, shape=None):
    if shape is None:
        return length,
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def keys_as_sorted_list(_dict):
    return sorted((list(_dict.keys())))


def values_as_sorted_list(_dict):
    return [_dict[k] for k in keys_as_sorted_list(_dict)]


def mlp(x, hidden_sizes, activation=nn.Tanh, ):
    return
