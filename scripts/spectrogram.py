#!/usr/bin/env python
"""
sound_meter.py

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import tempfile

import numpy as np
import rospy

import Gnuplot

import common

class SpectrogramPlotter:
    def __init__(self, channel_id=0):
        # init gnuplot
        self.gplot = Gnuplot.Gnuplot()

        self.gplot('unset border')
        self.gplot('unset xtics')
        self.gplot('unset ytics')
        self.gplot('set cbrange [-100:-30]')
        #self.gplot('unset colorbox')
        #self.gplot('set palette gray negative')
        self.gplot('set palette defined (0 "#ffffcc", 1 "#ffeda0", 2 "#fed976", 3 "#feb24c", 4 "#fd8d3c", 5 "#fc4e2a", 6 "#e31a1c", 7 "#bd0026", 8 "#800026")')

        self.tfile = tempfile.mkstemp()[1]
        self.cid = channel_id

    def __call__(self, ts, tf, fs=None, fbins=None):
        spec = 10.0 * np.log10(np.abs(tf[self.cid]) ** 2.0 + 1e-20)

        Gnuplot.GridData(spec, binary=True, filename=self.tfile)
        self.gplot('plot "%s" binary matrix with image' % self.tfile)

def main(topic):
    rospy.init_node('audio_meter', anonymous=True)
    ss = common.StftSegmenter(topic, SpectrogramPlotter(), 4,
                              2048, 1024, 100, 4, fs=48000,
                              min_freq=100, max_freq=8000)
    rospy.spin()
    ss.stop()

if __name__ == '__main__':
    main('/naoqi_driver_node/audio')

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

