"""
common.py

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

from collections import deque
import threading
import Queue

import numpy as np

import rospy
from naoqi_bridge_msgs.msg import AudioBuffer

import apkit

SAMPLE_DTYPE = np.int16

QUEUE_SIZE = 100

class StftSegmenter:
    """Apply STFT to audio stream and segment into blocks"""

    def __init__(self, audio_topic, callback, nch, win_size, hop_size,
                 block_size, block_hop, fs=None, min_freq=0, max_freq=-1,
                 ctx_frames=0, ahd_frames=0, amplify=0.0):
        """
        Args:
            audio_topic : audio stream topic name
            callback    : callback function once a block of stft is ready
            nch         : number of channels
            win_size    : window size
            hop_size    : hop size
            block_size  : number of frames in one block
            block_hop   : number of frame shifts between blocks
            fs          : sample rate, needed for frequency cropping
            min_freq    : mininum frequency, needed for frequency cropping
            max_freq    : maximum frequency, needed for frequency cropping
            ctx_frames  : (default 0) number of context frames in the past
            ahd_frames  : (default 0) number of context frames to look ahead
            amplify     : (default 0) amplify input power in dB
        """
        assert hop_size <= win_size     # TODO: hop more than a frame
        assert block_hop <= block_size

        # init parameters
        self.callback = callback
        self.nch = nch
        self.win_size = win_size
        self.hop_size = hop_size
        self.block_size = block_size + ctx_frames + ahd_frames
        self.ctx_frames = ctx_frames
        self.block_hop = block_hop
        self.fs = fs
        self.input_scale = np.power(10.0, amplify / 20.0)

        if fs is not None:
            self.min_fbin = min_freq * win_size / fs
            if max_freq >= 0:
                self.max_fbin = max_freq * win_size / fs
            else:
                self.max_fbin = win_size / 2
        else:
            self.min_fbin = 0
            self.max_fbin = win_size / 2

        self.fbins = np.fft.fftfreq(win_size)[self.min_fbin:self.max_fbin]

        # init buffers
        self.raw_buf = np.zeros((nch, win_size))
        self.raw_bufi = 0
        self.stft_buf = np.zeros((nch, self.block_size,
                                  self.max_fbin - self.min_fbin),
                                 dtype=np.complex128)
        self.stft_ts = deque()
        self.stft_bufi = 0
        self.blocks = Queue.Queue(maxsize=QUEUE_SIZE)

        # start a thread to get blocks form queue and invoke callback
        self.thread = threading.Thread(target=self.proc_blocks)
        self.thread.start()

        # subscribe to audio stream
        rospy.Subscriber(audio_topic, AudioBuffer,
                         self.receive_audio_buffer)

    def stop(self):
        try:
            while True:
                self.blocks.get_nowait()
        except Queue.Empty:
            pass
        self.blocks.put(None)

    def receive_audio_buffer(self, msg):
        assert len(msg.channelMap) == self.nch
        assert self.fs is None or msg.frequency == self.fs

        # msg time stamp
        ts = msg.header.stamp

        # deinterleave raw audio data
        data = np.array(msg.data, dtype=SAMPLE_DTYPE).reshape((-1, self.nch))

        # convert to float
        data = data.astype(float) / abs(np.iinfo(SAMPLE_DTYPE).min)

        # amplify input
        data *= self.input_scale

        while len(data) > 0:
            # length of data to push to buffer
            plen = min(len(data), self.win_size - self.raw_bufi)

            # push to buffer
            self.raw_buf[:,self.raw_bufi:self.raw_bufi+plen] = data[:plen].T
            self.raw_bufi += plen

            # remove used data
            data = data[plen:]

            # update time stamp of buffer end
            ts += rospy.Duration.from_sec(float(plen) / msg.frequency)

            # if buffer is full
            if self.raw_bufi == self.win_size:
                self.push_frame(ts)

                # shift to the left by hop_size
                self.raw_buf = np.roll(self.raw_buf, -self.hop_size, axis=1)
                self.raw_bufi -= self.hop_size

    def push_frame(self, ts):
        # compute stft of current frame
        self.stft_buf[:,self.stft_bufi] = \
                apkit.stft(self.raw_buf, apkit.cola_hamming, self.win_size,
                           self.hop_size, last_sample=True)[:,0,self.min_fbin:self.max_fbin]
        self.stft_bufi += 1
        self.stft_ts.append(ts)

        # if buffer is full
        if self.stft_bufi == self.block_size:
            try:
                self.blocks.put_nowait((self.stft_ts[self.ctx_frames],
                                        self.stft_buf))
            except Queue.Full:
                pass

            # shift to the left by block_hop
            self.stft_buf = np.roll(self.stft_buf, -self.block_hop, axis=1)
            self.stft_bufi -= self.block_hop
            for _ in xrange(self.block_hop):
                self.stft_ts.popleft()

    def proc_blocks(self):
        block = self.blocks.get()
        while block is not None:
            ts, tf = block
            self.callback(ts, tf, self.fs, self.fbins)
            block = self.blocks.get()

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

