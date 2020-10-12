"""
ssl_methods.py

Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import math
import abc

import numpy as np

import torch
from torch.autograd import Variable

import rospy
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Vector3
from visualization_msgs.msg import Marker
from perception_msgs.msg import SoundSource, SoundSourceArray

import apkit

Gnuplot = None

class SoundLocalizer:
    """Abstract class of frame-level sound localization methods"""

    __metaclass__  = abc.ABCMeta

    @abc.abstractmethod
    def candidate_doas(self):
        pass

    @abc.abstractmethod
    def compute_spatial_spec(self, tf, fs, fbins):
        pass

    def prepare_source(self, idx):
        x, y, z = self.doas[idx]
        res = SoundSource()
        res.direction.x = x
        res.direction.y = y
        res.direction.z = z
        res.level = 0.0
        return res

    def __init__(self, topic, frame_id, visualize=False, pub_marker=False,
                 marker_life=None, nb_size = 8.0 / 180 * math.pi, threshold=0.5):
        self.frame_id = frame_id
        self.visualize = visualize
        self.threshold = threshold

        # register publisher
        self.pub = rospy.Publisher(topic, SoundSourceArray,
                                   queue_size=10)

        if pub_marker:
            self.pub_vis = rospy.Publisher('visualization_marker', Marker, queue_size=10)
            self.vis_lifetime = rospy.Duration.from_sec(marker_life)
            self.marker_id = 0
        else:
            self.pub_vis = None

        # get candidate DOAs and find neigbors
        self.doas = self.candidate_doas()
        self.nlist = apkit.neighbor_list(self.doas, nb_size)

        if visualize:
            global Gnuplot
            if Gnuplot is None:
                import Gnuplot

            # init azimuth doa
            doa_azi = np.arctan2(self.doas[:,1], self.doas[:,0])
            index_doa_azi_sort = sorted(enumerate(doa_azi),
                                        key=lambda x: x[1])
            perm = [index for index, _ in index_doa_azi_sort]
            doa_azi_sort = [a for _, a in index_doa_azi_sort]

            # close circle
            perm.append(perm[0])
            doa_azi_sort.append(doa_azi_sort[0])

            self.azi_doas = doa_azi_sort
            self.perm = perm

            # init gnuplot
            self.gplot = Gnuplot.Gnuplot()

            self.gplot('set polar')
            self.gplot('unset border')
            self.gplot('unset margin')
            self.gplot('set tics scale 0')
            self.gplot('unset xtics')
            self.gplot('unset ytics')
            self.gplot('set rtics ("" 0, "" 0.25, "" 0.5, "" 0.75, "" 1.0)')
            self.gplot('unset raxis')
            self.gplot('set trange [-2*pi:2*pi]')
            self.gplot('set cbrange [0:1]')
            self.gplot('set grid polar pi/6')
            self.gplot('set size square')
            self.gplot('set key bm')
            self.gplot('set label at 1.2,0 "right" center rotate by -90 tc rgb "gray"')
            self.gplot('set label at -1.2,0 "left" center rotate by 90 tc rgb "gray"')
            self.gplot('set label at 0,1.2 "front" center tc rgb "gray"')
            self.gplot('set label at 0,-1.2 "rear" center tc rgb "gray"')
            #self.gplot('set palette defined (0 "#4d9221", 1 "#7fbc41", 2 "#b8e186", 3 "#e6f5d0", 4 "#f7f7f7", 5 "#fde0ef", 6 "#f1b6da", 7 "#de77ae", 8 "#c51b7d")')
            self.gplot("set palette defined ( 0 '#FFFFE5', 1 '#F7FCB9', 2 '#D9F0A3', 3 '#ADDD8E', 4 '#78C679', 5 '#41AB5D', 6 '#238443', 7 '#005A32' )")

    def __call__(self, ts, tf, fs, fbins):
        y, z = self.compute_spatial_spec(tf, fs, fbins)

        # find peaks above certain threshold
        pred, = apkit.local_maxima(np.expand_dims(y, 1), self.nlist, th_phi=self.threshold)
        if z is not None:
            pred = [p for p in pred if y[p] * z[p] >= self.threshold]

        # prepare results
        res = SoundSourceArray()
        res.header.frame_id = self.frame_id
        res.header.stamp = ts
        res.data = [self.prepare_source(p) for p in pred]

        # publish results
        self.pub.publish(res)

        # publish markers
        if self.pub_vis is not None:
            for p in pred:
                marker = Marker(header=Header(stamp=ts,
                                              frame_id=self.frame_id),
                                ns='audio_perception.ssl',
                                id=self.marker_id,
                                type=Marker.SPHERE,
                                action=0,
                                scale=Vector3(0.1, 0.1, 0.1),
                                color=ColorRGBA(r=0.3, g=1.0, b=0.3, a=1.0),
                                lifetime=self.vis_lifetime,
                                frame_locked=True)
                pos = self.doas[p]
                marker.pose.position.x = pos[0]
                marker.pose.position.y = pos[1]
                marker.pose.position.z = pos[2]
                self.pub_vis.publish(marker)
                self.marker_id = (self.marker_id + 1) % 1000

        # visualize raw results
        if self.visualize:
            ssl_data = np.stack((self.azi_doas, y[self.perm]), axis=1)
            # sns_data = np.stack((self.azi_doas, z[self.perm]), axis=1)
            self.gplot.plot('1.1 w l lw 2 lc rgb "gray" notitle',
                            Gnuplot.Data(ssl_data, using='($1+0.5*pi):2',
                                         with_='l lw 4 lc rgb "blue"',
                                         title='DOA likelihood'))

class NNSoundLocalizer(SoundLocalizer):
    """Sound localizer using neural network"""

    def __init__(self, net, extract_ft, topic, frame_id, gpu=True,
                 visualize=False, pub_marker=False, marker_life=None,
                 nb_size=8.0 / 180 * math.pi, threshold=0.5,
                 half_circle=False):
        # should be before calling super init
        self.half_circle = half_circle

        super(NNSoundLocalizer, self).__init__(topic, frame_id, visualize,
                                               pub_marker, marker_life,
                                               nb_size, threshold)
        net.eval()
        if gpu:
            net.cuda()
        else:
            net.cpu()

        self.net = net
        self.extract_ft = extract_ft
        self.gpu = gpu

    def candidate_doas(self):
        # load DOAs
        if self.half_circle:
            return apkit.load_pts_horizontal(181, half=True)
        else:
            return apkit.load_pts_horizontal()

    def compute_spatial_spec(self, tf, fs, fbins):
        # extract feature
        feat = self.extract_ft(tf)

        # run nn
        x = Variable(torch.unsqueeze(torch.from_numpy(feat), 0))
        if self.gpu:
            x = x.cuda()
            y = self.net(x).cpu()
        else:
            y = self.net(x)

        # ssl + sns
        y = y.data.numpy()
        if y.ndim == 3 and y.shape[1] == 2:
            z = y[0,1]
            y = y[0,0]
        else:
            y = y.flatten()
            z = np.ones(len(y)) * 1.0

        return y, z

class SpatialSpectrumSoundLocalizer(SoundLocalizer):
    """Sound localizer with spatial spectrum"""

    def __init__(self, sfunc, mic_pos, fs, topic, frame_id, visualize=False,
                 pub_marker=False, marker_life=None,
                 nb_size = 8.0 / 180 * math.pi, threshold=0.5):
        super(SpatialSpectrumSoundLocalizer, self).__init__(
                topic, frame_id, visualize, pub_marker, marker_life,
                nb_size, threshold)
        self.sfunc = sfunc
        self.fs = fs
        self.delays = apkit.compute_delay(mic_pos, self.candidate_doas(), fs=fs)

    def candidate_doas(self):
        return apkit.load_pts_horizontal()

    def compute_spatial_spec(self, tf, fs, fbins):
        assert fs == self.fs

        # compute empirical covariance matrix
        ecov = apkit.empirical_cov_mat(tf)
        nch, _, nblock, nfbin = ecov.shape

        phi = self.sfunc(ecov, self.delays, fbins)

        # max pooling
        return np.max(phi, axis=1), None

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

