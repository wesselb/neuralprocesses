import lab as B
from lab.shape import Dimension
from plum import Tuple

from itertools import accumulate
import numpy as np
import torch

from .dist import AbstractDistribution
from .. import _dispatch

__all__ = ["Mixture"]


class Mixture(AbstractDistribution):

    @_dispatch
    def __init__(self, *components: AbstractDistribution):
        weights = [1. / len(components) for _ in range(len(components))]
        self.__init__(weights, *components)

    @_dispatch
    def __init__(self, weights, *components: AbstractDistribution):
        self.weights = weights
        self.components = components

    def sample(self, state, dtype, *shape):
        
        samples = []
        w = B.stack(*[B.cast(B.dtype_float(dtype), w) for w in self.weights])
        
        for c in self.components:
            state, sample = c.sample(state, dtype, *shape) 
            samples.append(sample)
            
        samples = B.stack(*samples, axis=0)
        
        state, choices = B.randcat(state, w, *shape)
        indicators = B.stack(*[B.cast(dtype, choices == i) for i in range(B.shape(w, 0))], axis=0)
        
        indicators = B.expand_dims(indicators, axis=-1, times=B.rank(samples) - B.rank(indicators))
        samples = B.sum(indicators * samples, axis=0)
            
        return state, samples
