#!/usr/bin/env python
"""
sound_meter.py

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import numpy as np
import rospy

import apkit

import common

def print_sound_level(ts, tf, fs=None, fbins=None):
    print ts, 10.0 * np.log10(apkit.power_tf(tf))

def main(topic):
    rospy.init_node('audio_meter', anonymous=True)
    ss = common.StftSegmenter(topic, print_sound_level, 4, 2048, 1024, 7,
                              4, fs=48000, min_freq=100, max_freq=8000)
    rospy.spin()
    ss.stop()

if __name__ == '__main__':
    main('/naoqi_driver_node/audio')

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

