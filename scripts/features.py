"""
features.py

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import numpy as np

class StftAsReal:
    def __call__(self, tf):
        return np.concatenate((tf.real, tf.imag), axis=0).astype(np.float32)

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

