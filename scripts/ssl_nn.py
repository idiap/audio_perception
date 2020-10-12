#!/usr/bin/env python
"""
ssl_nn.py

Copyright (c) 2017, 2018 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import argparse

import rospy

import archs

import common
import features
from ssl_methods import NNSoundLocalizer

_N_CHANNEL = 4
_SAMPLE_RATE = 48000

def main(model, audio_topic, ssl_topic, extract_ft, win_size, hop_size,
         block_size, block_hop, ctx_frames, ahd_frames, min_freq,
         max_freq, visualize, frame_id, pub_marker, cpu, amplify):
    # init net
    net = archs.load_module(model)

    # init function
    sl = NNSoundLocalizer(net, extract_ft, ssl_topic, frame_id, gpu=not cpu,
                          visualize=visualize, pub_marker=pub_marker,
                          marker_life=1.05 * hop_size * block_hop / _SAMPLE_RATE)

    # init ros node
    rospy.init_node('ssl_nn', anonymous=True)

    # init stft segmenter
    ss = common.StftSegmenter(audio_topic, sl, _N_CHANNEL, win_size,
                              hop_size, block_size, block_hop,
                              fs=_SAMPLE_RATE, min_freq=min_freq,
                              max_freq=max_freq, ctx_frames=ctx_frames,
                              ahd_frames=ahd_frames, amplify=amplify)

    # run
    rospy.spin()

    # close
    ss.stop()

_FEATURES = {'stft' : features.StftAsReal}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sound source localization with neural network')
    parser.add_argument('model', metavar='MODEL_PATH', type=str,
                        help='path to trained model')
    parser.add_argument('--audio-topic', metavar='IN_TOPIC', type=str,
                        default='/naoqi_driver_node/audio',
                        help='audio stream topic name')
    parser.add_argument('--ssl-topic', metavar='OUT_TOPIC', type=str,
                        default='audio_perception/ssl',
                        help='audio stream topic name')
    parser.add_argument('-n', '--feature', metavar='FEATURE', type=str,
                        required=True, choices=_FEATURES.keys(),
                        help='feature extraction method')
    parser.add_argument('-w', '--window-size', metavar='WIN_SIZE',
                        type=int, default=2048,
                        help='(default 2048) analysis window size')
    parser.add_argument('-o', '--hop-size', metavar='HOP_SIZE', type=int,
                        default=1024,
                        help='(default 1024) hop size, number of samples between windows')
    parser.add_argument('--block-size', metavar='BLOCK_SIZE', type=int,
                        default=7, help='(default 7) number of frames per block')
    parser.add_argument('--block-hop', metavar='BLOCK_HOP', type=int,
                        default=4, help='(default 4) number of frames between consecutive blocks')
    parser.add_argument('--context-frames', metavar='N_CTX', type=int,
                        default=0, help='number of frames of context')
    parser.add_argument('--ahead-frames', metavar='N_AHD', type=int,
                        default=0, help='number of frames to look ahead')
    parser.add_argument('--min-freq', metavar='MIN_FREQ', type=int,
                        default=100, help='(default 100) minimum frequecny')
    parser.add_argument('--max-freq', metavar='MAX_FREQ', type=int,
                        default=8000, help='(default 8000) maximum frequecny')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='visualize results')
    parser.add_argument('--frame-id', metavar='FRAME_ID', type=str,
                        default='MicrophoneArray_frame',
                        help='reference frame (microphone array) in ROS')
    parser.add_argument('--marker', action='store_true',
                        help='publish visualize markers')
    parser.add_argument('--cpu', action='store_true',
                        help='use cpu (instead of gpu) for computation')
    parser.add_argument('--amplify', type=float, default=0.0,
                        help='(default 0) amplify input as dB in power')
    args = parser.parse_args(rospy.myargv()[1:])

    extract_ft = _FEATURES[args.feature]()

    main(args.model, args.audio_topic, args.ssl_topic, extract_ft,
         args.window_size, args.hop_size, args.block_size,
         args.block_hop, args.context_frames, args.ahead_frames,
         args.min_freq, args.max_freq, args.visualize, args.frame_id,
         args.marker, args.cpu, args.amplify)

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

