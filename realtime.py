from __future__ import division
'''
Copyright 2022 Joshuah Rainstar
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
'''
Copyright 2022 Joshuah Rainstar
This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''
#additional code contributed by Justin Engel
#idea and code and bugs by Joshuah Rainstar, Oscar Steila
#https://groups.io/g/NextGenSDRs/topic/spectral_denoising
#realtime.py 1.2.1 - 2/7/23 -  added back ssqueezepy. Since we dont bother with the EXE, because VST is being worked on, this is fine.
#the candid benefit is not much to negligable. However, this ensures that during masking and processing we are working with the mathematically most
#correct positioning of the DFT, per https://dsp.stackexchange.com/a/72590/50076, and therefore our error is reduced.
#this required a one-line modification to pyroomacoustic's DFT class in order to be compatible.
#implementing the changes during STFT formation proved too complex due to different approaches being used.
#new denoising filter!

#How to use this file:
#you will need 1 virtual audio cable- try https://vb-audio.com/Cable/ if you use windows.
#install and configure the virtual audio cable and your speakers for 16 bits, 48000hz, two channels.
#if you use OSX or Linux, you will have to modify this file by changing audio devices, settings and libraries appropriately.
#step one: put this file somewhere and remember where it is.
#step two: using https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Windows-x86_64.exe
#install python.
#step three: locate the dedicated python terminal in your start menu, called mambaforge prompt.
#within that prompt, give the following instructions:
#conda install pip numpy scipy
#pip install pipwin dearpygui pyroomacoustics ssqueezepy pyfftw
#pipwin install pyaudio
#if all of these steps successfully complete, you're ready to go, otherwise fix things.
#step four: set the output for your SDR software to the input for the cable device.
#virtual cable devices are loopback- their input is a speaker and their output is a mic.
#step five: on windows 10 or 11, go to settings -> system -> sound.
# select "app volume and speaker preferences" at the bottom and leave this window open.
# within the mambaforge prompt, navigate to the directory where you saved this file.
#run it with python cleanup.py
#within the settings window you opened before, a list of running programs will be visible.
#configure the input for the python program in that list to the output for the cable,
#and configure the output for the program in that list to your speakers.

#further recommendations:
#I recommend the use of a notch filter to reduce false positives and carrier signals disrupting the entropy filter.
#Please put other denoising methods except for the notch filter *after* this method.

import os
import numba
import numpy
os.environ['SSQ_PARALLEL'] = '0'

from ssqueezepy import stft as sstft
import pyroomacoustics as pra
from threading import Thread

import pyaudio
import dearpygui.dearpygui as dpg



import numpy as np
from numpy.lib.stride_tricks import as_strided as _as_strided

import warnings

import numpy as np
from numpy.fft import irfft, rfft

try:
    import pyfftw

    pyfftw_available = True
except ImportError:
    pyfftw_available = False

try:
    import mkl_fft  # https://github.com/IntelPython/mkl_fft

    mkl_available = True
except ImportError:
    mkl_available = False


class DFT(object):
    """
    Class for performing the Discrete Fourier Transform (DFT) of real signals.
    Attributes
    ----------
    X: numpy array
        Real DFT computed by ``analysis``.
    x: numpy array
        IDFT computed by ``synthesis``.
    nfft: int
        FFT size.
    D: int
        Number of channels.
    transform: str
        Which FFT package will be used.
    analysis_window: numpy array
        Window to be applied before DFT.
    synthesis_window: numpy array
        Window to be applied after inverse DFT.
    axis : int
        Axis over which to compute the FFT.
    precision, bits : string, np.float32, np.float64, np.complex64, np.complex128
        How many precision bits to use for the input. Twice the amount will be used
        for complex spectrum.
    Parameters
    ----------
    nfft: int
        FFT size.
    D: int, optional
        Number of channels. Default is 1.
    analysis_window: numpy array, optional
        Window to be applied before DFT. Default is no window.
    synthesis_window: numpy array, optional
        Window to be applied after inverse DFT. Default is no window.
    transform: str, optional
        which FFT package to use: ``numpy``, ``pyfftw``, or ``mkl``. Default is
        ``numpy``.
    axis : int, optional
        Axis over which to compute the FFT. Default is first axis.
    precision : string, np.float32, np.float64, np.complex64, np.complex128, optional
        How many precision bits to use for the input.
        If 'single'/np.float32/np.complex64, 32 bits for real inputs or 64 for complex spectrum.
        Otherwise, cast to 64 bits for real inputs or 128 for complex spectrum (default).
    """

    def __init__(
        self,
        nfft,
        D=1,
        analysis_window=None,
        synthesis_window=None,
        transform="numpy",
        axis=0,
        precision="double",
        bits=None,
    ):
        self.nfft = nfft
        self.D = D
        self.axis = axis

        if bits is not None and precision is not None:
            warnings.warn(
                'Deprecated keyword "bits" ignored in favor of new keyword "precision"',
                DeprecationWarning,
            )
        elif bits is not None and precision is None:
            warnings.warn(
                'Keyword "bits" is deprecated and has been replaced by "precision"'
            )
            if bits == 32:
                precision = "single"
            elif bits == 64:
                precision = "double"

        if (
            precision == np.float32
            or precision == np.complex64
            or precision == "single"
        ):
            time_dtype = np.float32
            freq_dtype = np.complex64
        else:
            time_dtype = np.float64
            freq_dtype = np.complex128

        if axis == 0:
            self.x = np.squeeze(np.zeros((self.nfft, self.D), dtype=time_dtype))
            self.X = np.squeeze(
                np.zeros((self.nfft // 2 + 1, self.D), dtype=freq_dtype)
            )
        elif axis == 1:
            self.x = np.squeeze(np.zeros((self.D, self.nfft), dtype=time_dtype))
            self.X = np.squeeze(
                np.zeros((self.D, self.nfft // 2 + 1), dtype=freq_dtype)
            )
        else:
            raise ValueError("Invalid 'axis' option. Must be 0 or 1.")

        if analysis_window is not None:
            if axis == 0 and D > 1:
                self.analysis_window = analysis_window[:, np.newaxis].astype(time_dtype)
            else:
                self.analysis_window = analysis_window.astype(time_dtype)
        else:
            self.analysis_window = None

        if synthesis_window is not None:
            if axis == 0 and D > 1:
                self.synthesis_window = synthesis_window[:, np.newaxis].astype(
                    time_dtype
                )
            else:
                self.synthesis_window = synthesis_window.astype(time_dtype)
        else:
            self.synthesis_window = None

        if transform == "fftw":
            if pyfftw_available:
                from pyfftw import FFTW, empty_aligned

                self.transform = transform
                # allocate input (real) and output for pyfftw
                if self.D == 1:
                    self.a = empty_aligned(self.nfft, dtype=time_dtype)
                    self.b = empty_aligned(self.nfft // 2 + 1, dtype=freq_dtype)
                    self._forward = FFTW(self.a, self.b)
                    self._backward = FFTW(self.b, self.a, direction="FFTW_BACKWARD")
                else:
                    if axis == 0:
                        self.a = empty_aligned([self.nfft, self.D], dtype=time_dtype)
                        self.b = empty_aligned(
                            [self.nfft // 2 + 1, self.D], dtype=freq_dtype
                        )
                    elif axis == 1:
                        self.a = empty_aligned([self.D, self.nfft], dtype=time_dtype)
                        self.b = empty_aligned(
                            [self.D, self.nfft // 2 + 1], dtype=freq_dtype
                        )
                    self._forward = FFTW(self.a, self.b, axes=(self.axis,))
                    self._backward = FFTW(
                        self.b, self.a, axes=(self.axis,), direction="FFTW_BACKWARD"
                    )
            else:
                warnings.warn(
                    "Could not import pyfftw wrapper for fftw functions. Using numpy's rfft instead."
                )
                self.transform = "numpy"
        elif transform == "mkl":
            if mkl_available:
                import mkl_fft

                self.transform = "mkl"
            else:
                warnings.warn(
                    "Could not import mkl wrapper. Using numpy's rfft instead."
                )
                self.transform = "numpy"
        else:
            self.transform = "numpy"

    def analysis(self, x):
        """
        Perform frequency analysis of a real input using DFT.
        Parameters
        ----------
        x : numpy array
            Real signal in time domain.
        Returns
        -------
        numpy array
            DFT of input.
        """

        # check for valid input
        if x.shape != self.x.shape:
            raise ValueError(
                "Invalid input dimensions! Got (%d, %d), expecting (%d, %d)."
                % (x.shape[0], x.shape[1], self.x.shape[0], self.x.shape[1])
            )

        # apply window if needed
        if self.analysis_window is not None:
            np.multiply(self.analysis_window, x, x)

        # apply DFT
        if self.transform == "fftw":
            self.a[:,] = x
            self.X[:,] = self._forward()
        elif self.transform == "mkl":
            self.X[:,] = mkl_fft.rfft_numpy(x, self.nfft, axis=self.axis)
        else:
            self.X[:,] = rfft(x, self.nfft, axis=self.axis)

        return self.X

    def synthesis(self, X=None):
        """
        Perform time synthesis of frequency domain to real signal using the
        inverse DFT.
        Parameters
        ----------
        X : numpy array, optional
            Complex signal in frequency domain. Default is to use DFT computed
            from ``analysis``.
        Returns
        -------
        numpy array
            IDFT of ``self.X`` or input if given.
        """

        # check for valid input
        if X is not None:
            if X.shape != self.X.shape:
                raise ValueError(
                    "Invalid input dimensions! Got (%d, %d), expecting (%d, %d)."
                    % (X.shape[0], X.shape[1], self.X.shape[0], self.X.shape[1])
                )

            self.X[:,] = X

        # inverse DFT
        if self.transform == "fftw":
            self.b[:] = self.X
            self.x[:,] = self._backward()
        elif self.transform == "mkl":
            self.x[:,] = mkl_fft.irfft_numpy(self.X, self.nfft, axis=self.axis)
        else:
            self.x[:,] = irfft(self.X, self.nfft, axis=self.axis)
        
        
        self.x = np.fft.fftshift(self.x, axes=0) #reverse the modulation from ssqueezepy
        # apply window if needed
        
        if self.synthesis_window is not None:
            np.multiply(self.synthesis_window, self.x, self.x)

        return self.x



class STFT(object):
    """
    A class for STFT processing.
    Parameters
    -----------
    N : int
        number of samples per frame
    hop : int
        hop size
    analysis_window : numpy array
        window applied to block before analysis
    synthesis_window : numpy array
        window applied to the block before synthesis
    channels : int
        number of signals
    transform : str, optional
        which FFT package to use: 'numpy' (default), 'pyfftw', or 'mkl'
    streaming : bool, optional
        whether (True, default) or not (False) to "stitch" samples between
        repeated calls of 'analysis' and 'synthesis' if we are receiving a
        continuous stream of samples.
    num_frames : int, optional
        Number of frames to be processed. If set, this will be strictly enforced
        as the STFT block will allocate memory accordingly. If not set, there
        will be no check on the number of frames sent to
        analysis/process/synthesis
        NOTE:
            1) num_frames = 0, corresponds to a "real-time" case in which each
            input block corresponds to [hop] samples.
            2) num_frames > 0, requires [(num_frames-1)*hop + N] samples as the
            last frame must contain [N] samples.
    precision : string, np.float32, np.float64, np.complex64, np.complex128, optional
        How many precision bits to use for the input.
        If 'single'/np.float32/np.complex64, 32 bits for real inputs or 64 for complex spectrum.
        Otherwise, cast to 64 bits for real inputs or 128 for complex spectrum (default).
    """

    def __init__(
        self,
        N,
        hop=None,
        analysis_window=None,
        synthesis_window=None,
        channels=1,
        transform="numpy",
        streaming=True,
        precision="double",
        **kwargs
    ):
        # initialize parameters
        self.num_samples = N  # number of samples per frame
        self.num_channels = channels  # number of channels
        self.mono = True if self.num_channels == 1 else False
        if hop is not None:  # hop size --> number of input samples
            self.hop = hop
        else:
            self.hop = self.num_samples

        if (
            precision == np.float32
            or precision == np.complex64
            or precision == "single"
        ):
            self.time_dtype = np.float32
            self.freq_dtype = np.complex64
        else:
            self.time_dtype = np.float64
            self.freq_dtype = np.complex128

        # analysis and synthesis window
        self.analysis_window = analysis_window
        self.synthesis_window = synthesis_window

        # prepare variables for DFT object
        self.transform = transform
        self.nfft = self.num_samples  # differ when there is zero-padding
        self.nbin = self.nfft // 2 + 1

        # initialize filter + zero padding --> use set_filter
        self.zf = 0
        self.zb = 0
        self.H = None  # filter frequency spectrum
        self.H_multi = None  # for multiple frames

        # check keywords
        if "num_frames" in kwargs.keys():
            self.fixed_input = True
            num_frames = kwargs["num_frames"]
            if num_frames < 0:
                raise ValueError("num_frames must be non-negative!")
            self.num_frames = num_frames
        else:
            self.fixed_input = False
            self.num_frames = 0

        # allocate all the required buffers
        self.streaming = streaming
        self._make_buffers()

    def _make_buffers(self):
        """
        Allocate memory for internal buffers according to FFT size, number of
        channels, and number of frames.
        """

        # state variables
        self.n_state = self.num_samples - self.hop
        self.n_state_out = self.nfft - self.hop

        # make DFT object
        self.dft = DFT(
            nfft=self.nfft,
            D=self.num_channels,
            analysis_window=self.analysis_window,
            synthesis_window=self.synthesis_window,
            transform=self.transform,
        )
        """
        1D array for num_channels=1 as the FFTW package can only take 1D array 
        for 1D DFT.
        """
        if self.mono:
            # input buffer
            self.fft_in_buffer = np.zeros(self.nfft, dtype=self.time_dtype)
            # state buffer
            self.x_p = np.zeros(self.n_state, dtype=self.time_dtype)
            # prev reconstructed samples
            self.y_p = np.zeros(self.n_state_out, dtype=self.time_dtype)
            # output samples
            self.out = np.zeros(self.hop, dtype=self.time_dtype)
        else:
            # input buffer
            self.fft_in_buffer = np.zeros(
                (self.nfft, self.num_channels), dtype=self.time_dtype
            )
            # state buffer
            self.x_p = np.zeros(
                (self.n_state, self.num_channels), dtype=self.time_dtype
            )
            # prev reconstructed samples
            self.y_p = np.zeros(
                (self.n_state_out, self.num_channels), dtype=self.time_dtype
            )
            # output samples
            self.out = np.zeros((self.hop, self.num_channels), dtype=self.time_dtype)

        # useful views on the input buffer
        self.fft_in_state = self.fft_in_buffer[self.zf : self.zf + self.n_state,]
        self.fresh_samples = self.fft_in_buffer[
            self.zf + self.n_state : self.zf + self.n_state + self.hop,
        ]
        self.old_samples = self.fft_in_buffer[
            self.zf + self.hop : self.zf + self.hop + self.n_state,
        ]

        # if fixed number of frames to process
        if self.fixed_input:
            if self.num_frames == 0:
                if self.mono:
                    self.X = np.zeros(self.nbin, dtype=self.freq_dtype)
                else:
                    self.X = np.zeros(
                        (self.nbin, self.num_channels), dtype=self.freq_dtype
                    )
            else:
                self.X = np.squeeze(
                    np.zeros(
                        (self.num_frames, self.nbin, self.num_channels),
                        dtype=self.freq_dtype,
                    )
                )
                # DFT object for multiple frames
                self.dft_frames = DFT(
                    nfft=self.nfft,
                    D=self.num_frames,
                    analysis_window=self.analysis_window,
                    synthesis_window=self.synthesis_window,
                    transform=self.transform,
                )
        else:  # we will allocate these on-the-fly
            self.X = None
            self.dft_frames = None

    def reset(self):
        """
        Reset state variables. Necessary after changing or setting the filter
        or zero padding.
        """

        if self.mono:
            self.fft_in_buffer[:] = 0.0
            self.x_p[:] = 0.0
            self.y_p[:] = 0.0
            self.X[:] = 0.0
            self.out[:] = 0.0
        else:
            self.fft_in_buffer[:, :] = 0.0
            self.x_p[:, :] = 0.0
            self.y_p[:, :] = 0.0
            self.X[:, :] = 0.0
            self.out[:, :] = 0.0

    def zero_pad_front(self, zf):
        """
        Set zero-padding at beginning of frame.
        """
        self.zf = zf
        self.nfft = self.num_samples + self.zb + self.zf
        self.nbin = self.nfft // 2 + 1
        if self.analysis_window is not None:
            self.analysis_window = np.concatenate((np.zeros(zf), self.analysis_window))
        if self.synthesis_window is not None:
            self.synthesis_window = np.concatenate(
                (np.zeros(zf), self.synthesis_window)
            )

        # We need to reallocate buffers after changing zero padding
        self._make_buffers()

    def zero_pad_back(self, zb):
        """
        Set zero-padding at end of frame.
        """
        self.zb = zb
        self.nfft = self.num_samples + self.zb + self.zf
        self.nbin = self.nfft // 2 + 1
        if self.analysis_window is not None:
            self.analysis_window = np.concatenate((self.analysis_window, np.zeros(zb)))
        if self.synthesis_window is not None:
            self.synthesis_window = np.concatenate(
                (self.synthesis_window, np.zeros(zb))
            )

        # We need to reallocate buffers after changing zero padding
        self._make_buffers()

    def set_filter(self, coeff, zb=None, zf=None, freq=False):
        """
        Set time-domain FIR filter with appropriate zero-padding.
        Frequency spectrum of the filter is computed and set for the object.
        There is also a check for sufficient zero-padding.
        Parameters
        -----------
        coeff : numpy array
            Filter in time domain.
        zb : int
            Amount of zero-padding added to back/end of frame.
        zf : int
            Amount of zero-padding added to front/beginning of frame.
        freq : bool
            Whether or not given coefficients (coeff) are in the frequency
            domain.
        """
        # apply zero-padding
        if zb is not None:
            self.zero_pad_back(zb)
        if zf is not None:
            self.zero_pad_front(zf)
        if not freq:
            # compute filter magnitude and phase spectrum
            self.H = self.freq_dtype(np.fft.rfft(coeff, self.nfft, axis=0))

            # check for sufficient zero-padding
            if self.nfft < (self.num_samples + len(coeff) - 1):
                raise ValueError(
                    "Insufficient zero-padding for chosen number "
                    "of samples per frame (L) and filter length "
                    "(h). Require zero-padding such that new "
                    "length is at least (L+h-1)."
                )
        else:
            if len(coeff) != self.nbin:
                raise ValueError("Invalid length for frequency domain " "coefficients.")
            self.H = coeff

        # prepare filter if fixed input case
        if self.fixed_input:
            if self.num_channels == 1:
                self.H_multi = np.tile(self.H, (self.num_frames, 1))
            else:
                self.H_multi = np.tile(self.H, (self.num_frames, 1, 1))

    def analysis(self, x):
        """
        Parameters
        -----------
        x  : 2D numpy array, [samples, channels]
            Time-domain signal.
        """

        # ----check correct number of channels
        x_shape = x.shape
        if not self.mono:
            if len(x_shape) < 1:  # received mono
                raise ValueError(
                    "Received 1-channel signal. Expecting %d "
                    "channels." % self.num_channels
                )
            if x_shape[1] != self.num_channels:
                raise ValueError(
                    "Incorrect number of channels. Received %d, "
                    "expecting %d." % (x_shape[1], self.num_channels)
                )
        else:  # expecting mono
            if len(x_shape) > 1:  # received multi-channel
                raise ValueError(
                    "Received %d channels; expecting 1D mono " "signal." % x_shape[1]
                )

        # ----check number of frames
        if self.streaming:  # need integer multiple of hops
            if self.fixed_input:
                if x_shape[0] != self.num_frames * self.hop:
                    raise ValueError(
                        "Input must be of length %d; received %d "
                        "samples." % (self.num_frames * self.hop, x_shape[0])
                    )
            else:
                self.num_frames = int(np.ceil(x_shape[0] / self.hop))
                extra_samples = (self.num_frames * self.hop) - x_shape[0]
                if extra_samples:
                    if self.mono:
                        x = np.concatenate((x, np.zeros(extra_samples)))
                    else:
                        x = np.concatenate(
                            (x, np.zeros((extra_samples, self.num_channels)))
                        )

        # non-streaming
        # need at least num_samples for last frame
        # e.g.[hop|hop|...|hop|num_samples]
        else:
            if self.fixed_input:
                if x_shape[0] != (self.hop * (self.num_frames - 1) + self.num_samples):
                    raise ValueError(
                        "Input must be of length %d; received %d "
                        "samples."
                        % (
                            (self.hop * (self.num_frames - 1) + self.num_samples),
                            x_shape[0],
                        )
                    )
            else:
                if x_shape[0] < self.num_samples:
                    # raise ValueError('Not enough samples. Received %d; need \
                    #     at least %d.' % (x_shape[0],self.num_samples))
                    extra_samples = self.num_samples - x_shape[0]
                    if self.mono:
                        x = np.concatenate((x, np.zeros(extra_samples)))
                    else:
                        x = np.concatenate(
                            (x, np.zeros((extra_samples, self.num_channels)))
                        )
                    self.num_frames = 1
                else:
                    # calculate num_frames and append zeros if necessary
                    self.num_frames = int(
                        np.ceil((x_shape[0] - self.num_samples) / self.hop) + 1
                    )
                    extra_samples = (
                        (self.num_frames - 1) * self.hop + self.num_samples
                    ) - x_shape[0]
                    if extra_samples:
                        if self.mono:
                            x = np.concatenate((x, np.zeros(extra_samples)))
                        else:
                            x = np.concatenate(
                                (x, np.zeros((extra_samples, self.num_channels)))
                            )

        # ----allocate memory if necessary
        if not self.fixed_input:
            self.X = np.squeeze(
                np.zeros(
                    (self.num_frames, self.nbin, self.num_channels),
                    dtype=self.freq_dtype,
                )
            )
            self.dft_frames = DFT(
                nfft=self.nfft,
                D=self.num_frames,
                analysis_window=self.analysis_window,
                synthesis_window=self.synthesis_window,
                transform=self.transform,
            )

        # ----use appropriate function
        if self.streaming:
            self._analysis_streaming(x)
        else:
            self.reset()
            self._analysis_non_streaming(x)

        return self.X

    def _analysis_single(self, x_n):
        """
        Transform new samples to STFT domain for analysis.
        Parameters
        -----------
        x_n : numpy array
            [self.hop] new samples
        """

        # correct input size check in: dft.analysis()
        self.fresh_samples[:,] = x_n[:,]  # introduce new samples
        self.x_p[:,] = self.old_samples  # save next state

        # apply DFT to current frame
        self.X[:] = self.dft.analysis(self.fft_in_buffer)

        # shift backwards in the buffer the state
        self.fft_in_state[:,] = self.x_p[:,]

    def _analysis_streaming(self, x):
        """
        STFT analysis for streaming case in which we expect
        [num_frames*hop] samples
        """

        if self.num_frames == 1:
            self._analysis_single(x)
        else:
    

            n = 0
            for k in range(self.num_frames):

                # introduce new samples
                self.fresh_samples[:,] = x[n : n + self.hop,]
                # save next state
                self.x_p[:,] = self.old_samples

                # apply DFT to current frame
                self.X[k,] = self.dft.analysis(self.fft_in_buffer)

                # shift backwards in the buffer the state
                self.fft_in_state[:,] = self.x_p[:,]

                n += self.hop


    def _analysis_non_streaming(self, x):
        """
        STFT analysis for non-streaming case in which we expect
        [(num_frames-1)*hop+num_samples] samples
        """

        ## ----- STRIDED WAY
        new_strides = (x.strides[0], self.hop * x.strides[0])
        new_shape = (self.num_samples, self.num_frames)

        if not self.mono:
            for c in range(self.num_channels):
                y = _as_strided(x[:, c], shape=new_shape, strides=new_strides)
                y = np.concatenate(
                    (
                        np.zeros((self.zf, self.num_frames)),
                        y,
                        np.zeros((self.zb, self.num_frames)),
                    )
                )

                if self.num_frames == 1:
                    self.X[:, c] = self.dft_frames.analysis(y[:, 0]).T
                else:
                    self.X[:, :, c] = self.dft_frames.analysis(y).T
        else:
            y = _as_strided(x, shape=new_shape, strides=new_strides)
            y = np.concatenate(
                (
                    np.zeros((self.zf, self.num_frames)),
                    y,
                    np.zeros((self.zb, self.num_frames)),
                )
            )

            if self.num_frames == 1:
                self.X[:] = self.dft_frames.analysis(y[:, 0]).T
            else:
                self.X[:] = self.dft_frames.analysis(y).T

    def _check_input_frequency_dimensions(self, X):
        """
        Ensure that given frequency data is valid, i.e. number of channels and
        number of frequency bins.
        If fixed_input=True, ensure expected number of frames. Otherwise, infer
        from given data.
        Axis order of X should be : [frames, frequencies, channels]
        """

        # check number of frames and correct number of bins
        X_shape = X.shape
        if len(X_shape) == 1:  # single channel, one frame
            num_frames = 1
        elif len(X_shape) == 2 and not self.mono:  # multi-channel, one frame
            num_frames = 1
        elif len(X_shape) == 2 and self.mono:  # single channel, multiple frames
            num_frames = X_shape[0]
        elif len(X_shape) == 3 and not self.mono:  # multi-channel, multiple frames
            num_frames = X_shape[0]
        else:
            raise ValueError("Invalid input shape.")

        # check number of bins
        if num_frames == 1:
            if X_shape[0] != self.nbin:
                raise ValueError(
                    "Invalid number of frequency bins! Expecting "
                    "%d, got %d." % (self.nbin, X_shape[0])
                )
        else:
            if X_shape[1] != self.nbin:
                raise ValueError(
                    "Invalid number of frequency bins! Expecting"
                    " %d, got %d." % (self.nbin, X_shape[1])
                )

        # check number of frames, if fixed input size
        if self.fixed_input:
            if num_frames != self.num_frames:
                raise ValueError("Input must have %d frames!" % self.num_frames)
            self.X[:] = X  # reset if size is alright
        else:
            self.X = X
            self.num_frames = num_frames

        return self.X

    def process(self, X=None):
        """
        Parameters
        -----------
        X  : numpy array
            X can take on multiple shapes:
            1) (N,) if it is single channel and only one frame
            2) (N,D) if it is multi-channel and only one frame
            3) (F,N) if it is single channel but multiple frames
            4) (F,N,D) if it is multi-channel and multiple frames
        Returns
        -----------
        x_r : numpy array
            Reconstructed time-domain signal.
        """

        # check that there is filter
        if self.H is None:
            return

        if X is not None:
            self._check_input_frequency_dimensions(X)

        # use appropriate function
        if self.num_frames == 1:
            self._process_single()
        elif self.num_frames > 1:
            self._process_multiple()

        return self.X

    def _process_single(self):
        np.multiply(self.X, self.H, self.X)

    def _process_multiple(self):
        if not self.fixed_input:
            if self.mono:
                self.H_multi = np.tile(self.H, (self.num_frames, 1))
            else:
                self.H_multi = np.tile(self.H, (self.num_frames, 1, 1))

        np.multiply(self.X, self.H_multi, self.X)

    def synthesis(self, X=None):
        """
        Parameters
        -----------
        X  : numpy array of frequency content
            X can take on multiple shapes:
            1) (N,) if it is single channel and only one frame
            2) (N,D) if it is multi-channel and only one frame
            3) (F,N) if it is single channel but multiple frames
            4) (F,N,D) if it is multi-channel and multiple frames
            where:
            - F is the number of frames
            - N is the number of frequency bins
            - D is the number of channels
        Returns
        -----------
        x_r : numpy array
            Reconstructed time-domain signal.
        """

        if X is not None:
            self._check_input_frequency_dimensions(X)

        # use appropriate function
        if self.num_frames == 1:
            return self._synthesis_single()
        elif self.num_frames > 1:
            return self._synthesis_multiple()

    def _synthesis_single(self):
        """
        Transform to time domain and reconstruct output with overlap-and-add.
        Returns
        -------
        numpy array
            Reconstructed array of samples of length <self.hop>.
        """

        # apply IDFT to current frame
        self.dft.synthesis(self.X)

        return self._overlap_and_add()

    def _synthesis_multiple(self):
        """
        Apply STFT analysis to multiple frames.
        Returns
        -----------
        x_r : numpy array
            Recovered signal.
        """

        # synthesis + overlap and add
        if not self.mono:
            x_r = np.zeros(
                (self.num_frames * self.hop, self.num_channels), dtype=self.time_dtype
            )

            n = 0
            for f in range(self.num_frames):
                # apply IDFT to current frame and reconstruct output
                x_r[n : n + self.hop,] = self._overlap_and_add(
                    self.dft.synthesis(self.X[f, :, :])
                )
                n += self.hop

        else:
            x_r = np.zeros(self.num_frames * self.hop, dtype=self.time_dtype)

            # treat number of frames as the multiple channels for DFT
            if not self.fixed_input:
                self.dft_frames = DFT(
                    nfft=self.nfft,
                    D=self.num_frames,
                    analysis_window=self.analysis_window,
                    synthesis_window=self.synthesis_window,
                    transform=self.transform,
                )

            # back to time domain
            mx = self.dft_frames.synthesis(self.X.T)

            # overlap and add
            n = 0
            for f in range(self.num_frames):
                x_r[n : n + self.hop,] = self._overlap_and_add(mx[:, f])
                n += self.hop

        return x_r

    def _overlap_and_add(self, x=None):
        if x is None:
            x = self.dft.x

        self.out[:,] = x[0 : self.hop,]  # fresh output samples

        # add state from previous frames when overlap is used
        if self.n_state_out > 0:
            m = np.minimum(self.hop, self.n_state_out)
            self.out[:m,] += self.y_p[:m,]
            # update state variables
            self.y_p[: -self.hop,] = self.y_p[self.hop :,]  # shift out left
            self.y_p[-self.hop :,] = 0.0
            self.y_p[:,] += x[-self.n_state_out :,]

        return self.out

@numba.njit(numba.float64(numba.float64[:]))
def man(arr):
    med = numpy.nanmedian(arr[numpy.nonzero(arr)])
    return numpy.nanmedian(numpy.abs(arr - med))

@numba.njit(numba.float64(numba.float64[:]))
def atd(arr):
    x = numpy.square(numpy.abs(arr - man(arr)))
    return numpy.sqrt(numpy.nanmean(x))

@numba.njit(numba.float64(numba.float64[:]))
def threshold(data: numpy.ndarray):
 a = numpy.sqrt(numpy.nanmean(numpy.square(numpy.abs(data -numpy.nanmedian(numpy.abs(data - numpy.nanmedian(data[numpy.nonzero(data)]))))))) + numpy.nanmedian(data[numpy.nonzero(data)])
 return a

@numba.njit(numba.float64(numba.float64[:]))
def threshhold(arr):
  return (atd(arr)+ numpy.nanmedian(arr[numpy.nonzero(arr)])) 

def moving_average(x, w):
    return numpy.convolve(x, numpy.ones(w), 'same') / w

def smoothpadded(data: numpy.ndarray,n:float):
  o = numpy.pad(data, n*2, mode='median')
  return moving_average(o,n)[n*2: -n*2]

def numpy_convolve_filter_longways(data: numpy.ndarray,N:int,M:int):
  E = N*2
  d = numpy.pad(array=data,pad_width=((0,0),(E,E)),mode="constant")  
  b = numpy.ravel(d)  
  for all in range(M):
       b[:] = ( b[:]  + (numpy.convolve(b[:], numpy.ones(N),mode="same") / N)[:])/2
  return d[:,E:-E]
#after i got the padding right.. this retuns identical results in a fraction of the time!

def numpy_convolve_filter_topways(data: numpy.ndarray,N:int,M:int):
  E = N*2
  d = numpy.pad(array=data,pad_width=((E,E),(0,0)),mode="constant")  
  d = d.T.copy()  
  b = numpy.ravel(d)  
  for all in range(M):
       b[:] = ( b[:]  + (numpy.convolve(b[:], numpy.ones(N),mode="same")[:] / N)[:])/2
  d = d.T
  return d[E:-E:]

def generate_true_logistic(points):
    if points ==1:
      return 1
    if points ==2:
      return numpy.asarray([0.5,0.5])
    if points ==3:
      return numpy.asarray([0.2,0.6,0.2])

    fprint = numpy.linspace(0.0,1.0,points)
    fprint[1:-1]  /= 1 - fprint[1:-1]
    fprint[1:-1]  = numpy.log(fprint[1:-1])
    fprint[-1] = ((2*fprint[-2])  - fprint[-3]) 
    fprint[0] = -fprint[-1]
    return numpy.interp(fprint, (fprint[0], fprint[-1]),  (0, 1))


def generate_logit_window(size,sym =True):
  if size % 2:
    e = generate_true_logistic((size+1)//2)
    result = numpy.zeros(size)
    result[0:(size+1)//2] = e
    result[(size+1)//2:] = e[0:-1][::-1]
    return result
  else:
    e = generate_true_logistic(size//2)
    e = numpy.hstack((e,e[::-1]))
    return e


def generate_hann(M, sym=True):
    a = [0.5, 0.5]
    fac = numpy.linspace(-numpy.pi, numpy.pi, M)
    w = numpy.zeros(M)
    for k in range(len(a)):
        w += a[k] * numpy.cos(k * fac)
    return w

def generate_sawtooth_filter(size):
  if size ==1:
      return 1
  if size ==2:
      return numpy.asarray([0.5,0.5])
  if size ==3:
      return numpy.asarray([0.2,0.6,0.2])

  fprint = numpy.linspace(0.0,1.0,size)
  if size % 2:
    e = numpy.linspace(0.0,1.0,(size+1)//2)
    result = numpy.zeros(size)
    result[0:(size+1)//2] = e
    result[(size+1)//2:] = e[0:-1][::-1]
    return result
  else:
    e = numpy.linspace(0.0,1.0,(size+1)//2)
    e = numpy.hstack((e,e[::-1]))
    return e

wavelet_data = numpy.asarray([0.,0.14285714,0.28571429,0.42857143,0.57142857,0.71428571,0.85714286,1.,0.85714286,0.71428571,0.57142857,0.42857143,0.28571429,0.14285714,0.])
def sawtooth_filter(data):
    E= 15 *2
    working = numpy.pad(array=data,pad_width=((E,E),(E,E)),mode="constant")  
    #wavelet_data = generate_sawtooth_filter(15)
    working2 = working.flatten()
    working2 = numpy.convolve(working2, wavelet_data, mode='same')
    working = working2.reshape(working.shape)
    working = working/7
    return  working[E:-E:,E:-E:]

def denoise_probable(working: numpy.ndarray):
  weights = numpy.zeros_like(working)
  e =   numpy.random.normal(numpy.median(working),threshold(working.flatten()),(working.shape)) + (working-weights)
  smoothed = sawtooth_filter(e)
  weights = numpy.abs(working-smoothed) #save the difference

  smoothed =   numpy.random.normal(numpy.median(working),threshold(working.flatten()),(working.shape)) + (working/weights)
  smoothed = sawtooth_filter(smoothed)
  weights = numpy.abs(working-smoothed) + weights #save the difference

  smoothed =   numpy.random.normal(numpy.median(working),threshold(working.flatten()),(working.shape)) + (working/weights)
  smoothed = sawtooth_filter(smoothed)
  weights = numpy.abs(working-smoothed) + weights #save the difference

  smoothed =   numpy.random.normal(numpy.median(working),threshold(working.flatten()),(working.shape)) + (working/weights)
  smoothed = sawtooth_filter(smoothed)

  result = numpy.abs(working-smoothed)
  result= (result - numpy.nanmin(result)) /numpy.ptp(result)
  result = result * numpy.ptp(working)
  return numpy.minimum(result,working)

@numba.njit(numba.float64[:](numba.float64[:,:]))
def fast_entropy(data: numpy.ndarray):
   logit = numpy.asarray([0.,0.08507164,0.17014328,0.22147297,0.25905871,0.28917305,0.31461489,0.33688201,0.35687314,0.37517276,0.39218487,0.40820283,0.42344877,0.43809738,0.45229105,0.46614996,0.47977928,0.49327447,0.50672553,0.52022072,0.53385004,0.54770895,0.56190262,0.57655123,0.59179717,0.60781513,0.62482724,0.64312686,0.66311799,0.68538511,0.71082695,0.74094129,0.77852703,0.82985672,0.91492836,1.])
   #note: if you alter the number of bins, you need to regenerate this array. currently set to consider 36 bins
   entropy = numpy.zeros(data.shape[1],dtype=numpy.float64)
   for each in numba.prange(data.shape[1]):
      d = data[:,each]
      d = numpy.interp(d, (d[0], d[-1]), (0, +1))
      entropy[each] = 1 - numpy.corrcoef(d, logit)[0,1]
   return entropy


#def determine_entropy_maximum(size:int):
#     logit = generate_true_logistic(size)
#     d = numpy.zeros(size)
 #    d[-1] = 1
 #    return  1 - numpy.corrcoef(d,logit)[0,1] 

@numba.jit(numba.float64[:,:](numba.float64[:,:],numba.int64[:],numba.float64,numba.float64[:]))
def fast_peaks(stft_:numpy.ndarray,entropy:numpy.ndarray,thresh:numpy.float32,entropy_unmasked:numpy.ndarray):
    mask = numpy.zeros_like(stft_)
    for each in numba.prange(stft_.shape[1]):
        data = stft_[:,each].copy()
        if entropy[each] == 0:
            mask[0:36,each] =  0
            continue #skip the calculations for this row, it's masked already
        constant = atd(data) + man(data)  #by inlining the calls higher in the function, it only ever sees arrays of one size and shape, which optimizes the code
        test = entropy_unmasked[each]  / 0.6091672572096941 #currently set for 36 bins, if you change the number of bins, or use a dynamic setting,
        #replace this number with a call to determine_entropy_maximum - it will range from 0.50 to 0.75 depending on the size from 16-256
        test = abs(test - 1) 
        thresh1 = (thresh*test)
        if numpy.isnan(thresh1):
            thresh1 = constant #catch errors
        constant = (thresh1+constant)/2
        data[data<constant] = 0
        data[data>0] = 1
        mask[0:36,each] = data[:]
    return mask

@numba.njit(numba.int64(numba.int64[:]))
def longestConsecutive(nums: numpy.ndarray):
        streak = 0
        prevstreak = 0
        for num in range(nums.size):
          if nums[num] == 1:
            streak += 1
          if nums[num] == 0:
            prevstreak = max(streak,prevstreak)
            streak = 0
        return max(streak,prevstreak)

@numba.jit(numba.int64[:](numba.int64[:],numba.int64,numba.int64,numba.int64))
def process_row(a:numpy.ndarray, value:int, threshold, replace):
    first = 0
    end = 0
    index = 0
    while first < a.size:
        if a[first]==value:
            index = first
            while index<a.size and a[index]==value:
              index += 1
            end = index
            if end-first+1 < threshold:
              for i in range(first, end):
                a[i] = replace
            first = end
        else:
            index = first
            while index<a.size and a[index]!=value:
               index += 1
            first = index
    return a

import copy

def mask_generation(stft_vh:numpy.ndarray,stft_vl: numpy.ndarray,NBINS:int):

    #24000/256 = 93.75 hz per frequency bin.
    #a 4000 hz window(the largest for SSB is roughly 43 bins.
    #https://en.wikipedia.org/wiki/Voice_frequency
    #however, practically speaking, voice frequency cuts off just above 3400hz.
    #*most* SSB channels are constrained far below this.
    #to catch most voice activity on shortwave, we use the first 32 bins, or 3000hz.
    #we automatically set all other bins to the residue value.
    #reconstruction or upsampling of this reduced bandwidth signal is a different problem we dont solve here.
    
    
    
    residue = man(stft_vl[0:36,:].flatten())
    residue = (residue - numpy.nanmin(stft_vl)) / numpy.ptp(stft_vl)
    residue = residue/100 #the exact amount to reduce this to is unclear

    lettuce_euler_macaroni = 0.057
    stft_vs = numpy.sort(stft_vl[0:36,:],axis=0) #sort the array
    entropy_unmasked = fast_entropy(stft_vs)
    entropy_unmasked[numpy.isnan(entropy_unmasked)] = 0
    entropy = smoothpadded(entropy_unmasked,3).astype(dtype=numpy.float64)

    factor = numpy.max(entropy)

    if factor < lettuce_euler_macaroni: 
      return (stft_vh[:,64:128] * 1e-5)


    entropy[entropy<lettuce_euler_macaroni] = 0
    entropy[entropy>0] = 1
    entropy = entropy.astype(dtype=numpy.int64)

    entropy_minimal = entropy[64-32:128+32] #concluded 
    nbins = numpy.sum(entropy_minimal)
    maxstreak = longestConsecutive(entropy_minimal)
    if nbins<22 and maxstreak<16:
      return (stft_vh[:,64:128]  * 1e-5)
    #what this will do is simply reduce the amount of work slightly

    entropy = process_row(entropy,0,6,1)
    entropy = process_row(entropy,1,2,0)
    #remove anomalies

    mask=numpy.zeros_like(stft_vh)

    stft_vh1 = stft_vh[0:36,:]
    thresh = threshold(stft_vh1[stft_vh1>residue])/2
    stft_vh1 = denoise_probable(stft_vh1)

    stft_vh1 = sawtooth_filter(stft_vh1)

    mask[0:36,:] = fast_peaks(stft_vh1,entropy,thresh,entropy_unmasked)
    mask = sawtooth_filter(mask)
    mask2 = numpy_convolve_filter_longways(mask[:,(64-16):(128+16)],13,3)
    mask2 = numpy_convolve_filter_topways(mask2[:,16:-16],3,3) 
    mask2[mask2<residue] = residue

    return mask2


class FilterThread(Thread):
    def __init__(self):
        super(FilterThread, self).__init__()
        self.running = True
        self.NFFT = 512
        self.NBINS=36
        self.hop = 128
        self.hann = generate_hann(self.NFFT)
        self.logistic = generate_logit_window(self.NFFT)
        self.synthesis = pra.transform.stft.compute_synthesis_window(self.hann, self.hop)
        self.stft = STFT(512, hop=self.hop, analysis_window=self.hann,synthesis_window=self.synthesis ,online=True)

        self.audio = numpy.zeros(8192*3)

    def process(self,data,):        
           self.audio = numpy.roll(self.audio,-8192)
           self.audio[-8192:] = data[:]
           hann = sstft(self.audio,self.hann,512,hop_len=128,dtype= 'float64')
           logit = sstft(self.audio,self.logistic,512,hop_len=128,dtype= 'float64')
           mask = mask_generation(numpy.abs(hann),numpy.abs(logit),self.NBINS).T
           hann = hann[:,64:128]
           hann = hann.T
           result = hann * mask
           output = self.stft.synthesis(result)
           return output
  
  
    def stop(self):
        self.running = False 

        
def padarray(A, size):
    t = size - len(A)
    return numpy.pad(A, pad_width=(0, t), mode='constant',constant_values=numpy.std(A))

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def process_data(data: numpy.ndarray):
    print("processing ", data.size / rate, " seconds long file at ", rate, " rate.")
    start = time.time()
    filter = FilterThread()
    processed = []
    for each in chunks(data, 8192):
        if each.size == 8192:
            a = filter.process(each)
            processed.append(a)
        else:
            psize = each.size
            working = padarray(each, 8192)
            processed.append(filter.process(working)[0:psize])
    end = time.time()
    print("took ", end - start, " to process ", data.size / rate)
    return numpy.concatenate((processed), axis=0)     


class StreamSampler(object):

    def __init__(self, sample_rate=48000, channels=2,  micindex=1, speakerindex=1, dtype=numpy.float32):
        self.pa = pyaudio.PyAudio()
        self.processing_size = 8192
        self.sample_rate = sample_rate
        self.channels = channels
        self.rightthread = FilterThread()
        self.leftthread = FilterThread()
        self.micindex = micindex
        self.speakerindex = speakerindex
        self.micstream = None
        self.speakerstream = None
        self.speakerdevice = ""
        self.micdevice = ""
     
    

    def _update_streams(self):
        """Call if sample rate, channels, dtype, or something about the stream changes."""
        was_running = self.is_running()

        self.stop()
        self.micstream = None
        self.speakerstream = None
        if was_running:
            self.listen()

    def is_running(self):
        try:
            return self.micstream.is_active() or self.speakerstream.is_active()
        except (AttributeError, Exception):
            return False

    def stop(self):
        try:
            self.micstream.close()
        except (AttributeError, Exception):
            pass
        try:
            self.speakerstream.close()
        except (AttributeError, Exception):
            pass
        try:
            self.rightthread.join()
            self.leftthread.join()
        except (AttributeError, Exception):
            pass

    def open_mic_stream(self):
        device_index = None
        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)
            # print("Device %d: %s" % (i, devinfo["name"]))
            if devinfo['maxInputChannels'] == 2:
                for keyword in ["microsoft"]:
                    if keyword in devinfo["name"].lower():
                        self.micdevice = devinfo["name"]
                        device_index = i
                        self.micindex = device_index

        if device_index is None:
            print("No preferred input found; using default input device.")

        stream = self.pa.open(format=pyaudio.paFloat32,
                              channels=self.channels,
                              rate=int(self.sample_rate),
                              input=True,
                              input_device_index=self.micindex,  # device_index,
                              # each frame carries twice the data of the frames
                              frames_per_buffer=int(self.processing_size),
                              stream_callback=self.non_blocking_stream_read,
                              start=False  # Need start to be False if you don't want this to start right away
                              )

        return stream

    def open_speaker_stream(self):
        device_index = None
        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)
            # print("Device %d: %s" % (i, devinfo["name"]))
            if devinfo['maxOutputChannels'] == 2:
                for keyword in ["microsoft"]:
                    if keyword in devinfo["name"].lower():
                        self.speakerdevice = devinfo["name"]
                        device_index = i
                        self.speakerindex = device_index

        if device_index is None:
            print("No preferred output found; using default output device.")

        stream = self.pa.open(format=pyaudio.paFloat32,
                              channels=self.channels,
                              rate=int(self.sample_rate),
                              output=True,
                              output_device_index=self.speakerindex,
                              frames_per_buffer=int(self.processing_size),
                              start=False  # Need start to be False if you don't want this to start right away
                              )
        return stream

    def non_blocking_stream_read(self, in_data, frame_count, time_info, status):
        audio_in = numpy.ndarray(buffer=memoryview(in_data), dtype=numpy.float32,
                                            shape=[int(self.processing_size * self.channels)]).reshape(-1,
                                                                                                         self.channels)

        chans = []
        chans.append(self.rightthread.process(audio_in[:,0]))
        chans.append(self.leftthread.process(audio_in[:,1]))

        self.speakerstream.write(numpy.column_stack(chans).astype(numpy.float32).tobytes())
        return None, pyaudio.paContinue


    def stream_start(self):
        self.rightthread.start()
        self.leftthread.start()
        if self.micstream is None:
            self.micstream = self.open_mic_stream()
        self.micstream.start_stream()
        if self.speakerstream is None:
            self.speakerstream = self.open_speaker_stream()
        self.speakerstream.start_stream()
        # Don't do this here. Do it in main. Other things may want to run while this stream is running
        # while self.micstream.is_active():
        #     eval(input("main thread is now paused"))

    listen = stream_start  # Just set a new variable to the same method


if __name__ == "__main__":
    SS = StreamSampler()
    SS.listen()
    def close():
        dpg.destroy_context()
        SS.stop()
        quit()


    dpg.create_context()   

    dpg.create_viewport(title= 'Streamclean', height=100, width=400)
    dpg.setup_dearpygui()
    dpg.configure_app(auto_device=True)

    dpg.show_viewport()
    dpg.start_dearpygui()
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
    close()  # clean up the program runtime when the user closes the window
