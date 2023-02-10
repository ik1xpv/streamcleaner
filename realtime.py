from __future__ import division
'''
Copyright 2023 Joshuah Rainstar
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
'''
Copyright 2023 Joshuah Rainstar
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
#realtime.py 1.2.3 - 2/10/23 

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
#run it with python realtime.py
#within the settings window you opened before, a list of running programs will be visible.
#configure the input for the python program in that list to the output for the cable,
#and configure the output for the program in that list to your speakers.

#further recommendations:
#I recommend the use of a notch filter to reduce false positives and carrier signals disrupting the entropy filter.
#Please put other denoising methods except for the notch filter *after* this method.

import os
import numba
import numpy 
os.environ['SSQ_PARALLEL'] = '0' #avoids high cpu usage, we dont need maximum output

from ssqueezepy import stft as sstft
import pyroomacoustics as pra

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


def moving_average(x, w):
    return numpy.convolve(x, numpy.ones(w), 'same') / w

def smoothpadded(data: numpy.ndarray,n:float):
  o = numpy.pad(data, n*2, mode='constant')
  return moving_average(o,n)[n*2: -n*2]


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

sawtoothdata = numpy.asarray([0.,0.14285714,0.28571429,0.42857143,0.57142857,0.71428571,0.85714286,1.,0.85714286,0.71428571,0.57142857,0.42857143,0.28571429,0.14285714,0.])
def sawtooth_filter(data):
    E= 30
    working = numpy.pad(array=data,pad_width=((E,E),(E,E)),mode="constant")  
    #wavelet_data = generate_sawtooth_filter(15)
    working2 = working.flatten()
    working2 = numpy.convolve(working2, sawtoothdata, mode='same')
    working = working2.reshape(working.shape)
    working = working/7
    return  working[E:-E:,E:-E:]


def renormalize(data,previous):
  data = data - numpy.nanmin(data)
  data = data/numpy.ptp(data)
  data = data * numpy.ptp(previous)
  return data

def convolve_custom_filter_2d(data: numpy.ndarray,N:int,M:int,O:int):
  E = N*2
  F = M*2
  padded = numpy.pad(array=data,pad_width=((F,F),(E,E)),mode="constant")  
  normal = padded.copy()
  normal_t = padded.T.copy()
  b = numpy.ravel(normal)
  c = numpy.ravel(normal_t)

  for all in range(O):
      normal = padded.copy()
      normal_t = padded.T.copy()
      b = numpy.ravel(normal)
      c = numpy.ravel(normal_t)
      b[:] = (numpy.convolve(b[:], numpy.ones(N),mode="same") / N)[:]
      c[:] =  (numpy.convolve(c[:], numpy.ones(M),mode="same") / M)[:]
      padded = (normal + normal_t.T.copy())
  return renormalize(padded[F:-F,E:-E],data)

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
def remove_outliers(a:numpy.ndarray, value:int, threshold, replace):
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


class Filter(object):
    def __init__(self):
        self.NFFT = 512
        self.NBINS=36
        self.hop = 128
        self.hann = numpy.asarray([0.00000000e+00,3.77965773e-05,1.51180595e-04,3.40134910e-04 ,6.04630957e-04,9.44628746e-04,1.36007687e-03,1.85091253e-03 ,2.41706151e-03,3.05843822e-03,3.77494569e-03,4.56647559e-03 ,5.43290826e-03,6.37411270e-03,7.38994662e-03,8.48025644e-03 ,9.64487731e-03,1.08836332e-02,1.21963367e-02,1.35827895e-02 ,1.50427819e-02,1.65760932e-02,1.81824916e-02,1.98617342e-02 ,2.16135671e-02,2.34377255e-02,2.53339336e-02,2.73019047e-02 ,2.93413412e-02,3.14519350e-02,3.36333667e-02,3.58853068e-02 ,3.82074146e-02,4.05993391e-02,4.30607187e-02,4.55911813e-02 ,4.81903443e-02,5.08578147e-02,5.35931893e-02,5.63960544e-02 ,5.92659864e-02,6.22025514e-02,6.52053053e-02,6.82737943e-02 ,7.14075543e-02,7.46061116e-02,7.78689827e-02,8.11956742e-02 ,8.45856832e-02,8.80384971e-02,9.15535940e-02,9.51304424e-02 ,9.87685015e-02,1.02467221e-01,1.06226043e-01,1.10044397e-01 ,1.13921708e-01,1.17857388e-01,1.21850843e-01,1.25901469e-01 ,1.30008654e-01,1.34171776e-01,1.38390206e-01,1.42663307e-01 ,1.46990432e-01,1.51370928e-01,1.55804131e-01,1.60289372e-01 ,1.64825973e-01,1.69413247e-01,1.74050502e-01,1.78737036e-01 ,1.83472140e-01,1.88255099e-01,1.93085190e-01,1.97961681e-01 ,2.02883837e-01,2.07850913e-01,2.12862158e-01,2.17916814e-01 ,2.23014117e-01,2.28153297e-01,2.33333576e-01,2.38554171e-01 ,2.43814294e-01,2.49113148e-01,2.54449933e-01,2.59823842e-01 ,2.65234062e-01,2.70679775e-01,2.76160159e-01,2.81674384e-01 ,2.87221617e-01,2.92801019e-01,2.98411747e-01,3.04052952e-01 ,3.09723782e-01,3.15423378e-01,3.21150881e-01,3.26905422e-01 ,3.32686134e-01,3.38492141e-01,3.44322565e-01,3.50176526e-01 ,3.56053138e-01,3.61951513e-01,3.67870760e-01,3.73809982e-01 ,3.79768282e-01,3.85744760e-01,3.91738511e-01,3.97748631e-01 ,4.03774209e-01,4.09814335e-01,4.15868096e-01,4.21934577e-01 ,4.28012860e-01,4.34102027e-01,4.40201156e-01,4.46309327e-01 ,4.52425614e-01,4.58549094e-01,4.64678841e-01,4.70813928e-01 ,4.76953428e-01,4.83096412e-01,4.89241951e-01,4.95389117e-01 ,5.01536980e-01,5.07684611e-01,5.13831080e-01,5.19975458e-01 ,5.26116815e-01,5.32254225e-01,5.38386758e-01,5.44513487e-01 ,5.50633486e-01,5.56745831e-01,5.62849596e-01,5.68943859e-01 ,5.75027699e-01,5.81100196e-01,5.87160431e-01,5.93207489e-01 ,5.99240456e-01,6.05258418e-01,6.11260467e-01,6.17245695e-01 ,6.23213197e-01,6.29162070e-01,6.35091417e-01,6.41000339e-01 ,6.46887944e-01,6.52753341e-01,6.58595644e-01,6.64413970e-01 ,6.70207439e-01,6.75975174e-01,6.81716305e-01,6.87429962e-01 ,6.93115283e-01,6.98771407e-01,7.04397480e-01,7.09992651e-01 ,7.15556073e-01,7.21086907e-01,7.26584315e-01,7.32047467e-01 ,7.37475536e-01,7.42867702e-01,7.48223150e-01,7.53541070e-01 ,7.58820659e-01,7.64061117e-01,7.69261652e-01,7.74421479e-01 ,7.79539817e-01,7.84615893e-01,7.89648938e-01,7.94638193e-01 ,7.99582902e-01,8.04482319e-01,8.09335702e-01,8.14142317e-01 ,8.18901439e-01,8.23612347e-01,8.28274329e-01,8.32886681e-01 ,8.37448705e-01,8.41959711e-01,8.46419017e-01,8.50825950e-01 ,8.55179843e-01,8.59480037e-01,8.63725883e-01,8.67916738e-01 ,8.72051970e-01,8.76130952e-01,8.80153069e-01,8.84117711e-01 ,8.88024281e-01,8.91872186e-01,8.95660845e-01,8.99389686e-01 ,9.03058145e-01,9.06665667e-01,9.10211707e-01,9.13695728e-01 ,9.17117204e-01,9.20475618e-01,9.23770461e-01,9.27001237e-01 ,9.30167455e-01,9.33268638e-01,9.36304317e-01,9.39274033e-01 ,9.42177336e-01,9.45013788e-01,9.47782960e-01,9.50484434e-01 ,9.53117800e-01,9.55682662e-01,9.58178630e-01,9.60605328e-01 ,9.62962389e-01,9.65249456e-01,9.67466184e-01,9.69612237e-01 ,9.71687291e-01,9.73691033e-01,9.75623159e-01,9.77483377e-01 ,9.79271407e-01,9.80986977e-01,9.82629829e-01,9.84199713e-01 ,9.85696393e-01,9.87119643e-01,9.88469246e-01,9.89745000e-01 ,9.90946711e-01,9.92074198e-01,9.93127290e-01,9.94105827e-01 ,9.95009663e-01,9.95838660e-01,9.96592693e-01,9.97271648e-01 ,9.97875422e-01,9.98403924e-01,9.98857075e-01,9.99234805e-01 ,9.99537058e-01,9.99763787e-01,9.99914959e-01,9.99990551e-01 ,9.99990551e-01,9.99914959e-01,9.99763787e-01,9.99537058e-01 ,9.99234805e-01,9.98857075e-01,9.98403924e-01,9.97875422e-01 ,9.97271648e-01,9.96592693e-01,9.95838660e-01,9.95009663e-01 ,9.94105827e-01,9.93127290e-01,9.92074198e-01,9.90946711e-01 ,9.89745000e-01,9.88469246e-01,9.87119643e-01,9.85696393e-01 ,9.84199713e-01,9.82629829e-01,9.80986977e-01,9.79271407e-01 ,9.77483377e-01,9.75623159e-01,9.73691033e-01,9.71687291e-01 ,9.69612237e-01,9.67466184e-01,9.65249456e-01,9.62962389e-01 ,9.60605328e-01,9.58178630e-01,9.55682662e-01,9.53117800e-01 ,9.50484434e-01,9.47782960e-01,9.45013788e-01,9.42177336e-01 ,9.39274033e-01,9.36304317e-01,9.33268638e-01,9.30167455e-01 ,9.27001237e-01,9.23770461e-01,9.20475618e-01,9.17117204e-01 ,9.13695728e-01,9.10211707e-01,9.06665667e-01,9.03058145e-01 ,8.99389686e-01,8.95660845e-01,8.91872186e-01,8.88024281e-01 ,8.84117711e-01,8.80153069e-01,8.76130952e-01,8.72051970e-01 ,8.67916738e-01,8.63725883e-01,8.59480037e-01,8.55179843e-01 ,8.50825950e-01,8.46419017e-01,8.41959711e-01,8.37448705e-01 ,8.32886681e-01,8.28274329e-01,8.23612347e-01,8.18901439e-01 ,8.14142317e-01,8.09335702e-01,8.04482319e-01,7.99582902e-01 ,7.94638193e-01,7.89648938e-01,7.84615893e-01,7.79539817e-01 ,7.74421479e-01,7.69261652e-01,7.64061117e-01,7.58820659e-01 ,7.53541070e-01,7.48223150e-01,7.42867702e-01,7.37475536e-01 ,7.32047467e-01,7.26584315e-01,7.21086907e-01,7.15556073e-01 ,7.09992651e-01,7.04397480e-01,6.98771407e-01,6.93115283e-01 ,6.87429962e-01,6.81716305e-01,6.75975174e-01,6.70207439e-01 ,6.64413970e-01,6.58595644e-01,6.52753341e-01,6.46887944e-01 ,6.41000339e-01,6.35091417e-01,6.29162070e-01,6.23213197e-01 ,6.17245695e-01,6.11260467e-01,6.05258418e-01,5.99240456e-01 ,5.93207489e-01,5.87160431e-01,5.81100196e-01,5.75027699e-01 ,5.68943859e-01,5.62849596e-01,5.56745831e-01,5.50633486e-01 ,5.44513487e-01,5.38386758e-01,5.32254225e-01,5.26116815e-01 ,5.19975458e-01,5.13831080e-01,5.07684611e-01,5.01536980e-01 ,4.95389117e-01,4.89241951e-01,4.83096412e-01,4.76953428e-01 ,4.70813928e-01,4.64678841e-01,4.58549094e-01,4.52425614e-01 ,4.46309327e-01,4.40201156e-01,4.34102027e-01,4.28012860e-01 ,4.21934577e-01,4.15868096e-01,4.09814335e-01,4.03774209e-01 ,3.97748631e-01,3.91738511e-01,3.85744760e-01,3.79768282e-01 ,3.73809982e-01,3.67870760e-01,3.61951513e-01,3.56053138e-01 ,3.50176526e-01,3.44322565e-01,3.38492141e-01,3.32686134e-01 ,3.26905422e-01,3.21150881e-01,3.15423378e-01,3.09723782e-01 ,3.04052952e-01,2.98411747e-01,2.92801019e-01,2.87221617e-01 ,2.81674384e-01,2.76160159e-01,2.70679775e-01,2.65234062e-01 ,2.59823842e-01,2.54449933e-01,2.49113148e-01,2.43814294e-01 ,2.38554171e-01,2.33333576e-01,2.28153297e-01,2.23014117e-01 ,2.17916814e-01,2.12862158e-01,2.07850913e-01,2.02883837e-01 ,1.97961681e-01,1.93085190e-01,1.88255099e-01,1.83472140e-01 ,1.78737036e-01,1.74050502e-01,1.69413247e-01,1.64825973e-01 ,1.60289372e-01,1.55804131e-01,1.51370928e-01,1.46990432e-01 ,1.42663307e-01,1.38390206e-01,1.34171776e-01,1.30008654e-01 ,1.25901469e-01,1.21850843e-01,1.17857388e-01,1.13921708e-01 ,1.10044397e-01,1.06226043e-01,1.02467221e-01,9.87685015e-02 ,9.51304424e-02,9.15535940e-02,8.80384971e-02,8.45856832e-02 ,8.11956742e-02,7.78689827e-02,7.46061116e-02,7.14075543e-02 ,6.82737943e-02,6.52053053e-02,6.22025514e-02,5.92659864e-02 ,5.63960544e-02,5.35931893e-02,5.08578147e-02,4.81903443e-02 ,4.55911813e-02,4.30607187e-02,4.05993391e-02,3.82074146e-02 ,3.58853068e-02,3.36333667e-02,3.14519350e-02,2.93413412e-02 ,2.73019047e-02,2.53339336e-02,2.34377255e-02,2.16135671e-02 ,1.98617342e-02,1.81824916e-02,1.65760932e-02,1.50427819e-02 ,1.35827895e-02,1.21963367e-02,1.08836332e-02,9.64487731e-03 ,8.48025644e-03,7.38994662e-03,6.37411270e-03,5.43290826e-03 ,4.56647559e-03,3.77494569e-03,3.05843822e-03,2.41706151e-03 ,1.85091253e-03,1.36007687e-03,9.44628746e-04,6.04630957e-04 ,3.40134910e-04,1.51180595e-04,3.77965773e-05,0.00000000e+00])
        self.logistic = numpy.asarray([0.,0.05590667,0.11181333,0.14464919,0.16804013,0.18625637,0.20119997,0.21388557,0.22491881,0.23469034,0.24346692,0.2514388,0.25874646,0.26549659,0.27177213,0.27763882,0.28314967,0.28834802,0.2932698,0.29794509,0.30239936,0.30665433,0.31072869,0.31463866,0.31839837,0.32202023,0.32551518,0.32889293,0.33216214,0.33533054,0.3384051,0.34139207,0.34429715,0.34712548,0.34988176,0.35257028,0.35519495,0.35775939,0.36026692,0.36272059,0.36512323,0.36747746,0.36978573,0.37205028,0.37427323,0.37645655,0.37860207,0.38071152,0.3827865,0.38482855,0.38683907,0.38881941,0.39077084,0.39269455,0.39459167,0.39646327,0.39831036,0.40013389,0.40193478,0.4037139,0.40547206,0.40721004,0.4089286,0.41062845,0.41231027,0.4139747,0.41562237,0.41725387,0.41886977,0.42047061,0.42205693,0.42362923,0.42518798,0.42673365,0.4282667,0.42978755,0.43129661,0.43279429,0.43428097,0.43575703,0.43722282,0.43867871,0.44012502,0.44156207,0.4429902,0.44440971,0.44582088,0.44722402,0.44861941,0.45000731,0.451388,0.45276174,0.45412877,0.45548934,0.4568437,0.45819207,0.45953469,0.46087178,0.46220355,0.46353023,0.46485202,0.46616913,0.46748176,0.46879011,0.47009437,0.47139473,0.47269138,0.4739845,0.47527428,0.4765609,0.47784452,0.47912534,0.48040351,0.48167921,0.48295261,0.48422387,0.48549315,0.48676063,0.48802646,0.4892908,0.49055382,0.49181567,0.4930765,0.49433648,0.49559576,0.4968545,0.49811286,0.49937098,0.50062902,0.50188714,0.5031455,0.50440424,0.50566352,0.5069235,0.50818433,0.50944618,0.5107092,0.51197354,0.51323937,0.51450685,0.51577613,0.51704739,0.51832079,0.51959649,0.52087466,0.52215548,0.5234391,0.52472572,0.5260155,0.52730862,0.52860527,0.52990563,0.53120989,0.53251824,0.53383087,0.53514798,0.53646977,0.53779645,0.53912822,0.54046531,0.54180793,0.5431563,0.54451066,0.54587123,0.54723826,0.548612,0.54999269,0.55138059,0.55277598,0.55417912,0.55559029,0.5570098,0.55843793,0.55987498,0.56132129,0.56277718,0.56424297,0.56571903,0.56720571,0.56870339,0.57021245,0.5717333,0.57326635,0.57481202,0.57637077,0.57794307,0.57952939,0.58113023,0.58274613,0.58437763,0.5860253,0.58768973,0.58937155,0.5910714,0.59278996,0.59452794,0.5962861,0.59806522,0.59986611,0.60168964,0.60353673,0.60540833,0.60730545,0.60922916,0.61118059,0.61316093,0.61517145,0.6172135,0.61928848,0.62139793,0.62354345,0.62572677,0.62794972,0.63021427,0.63252254,0.63487677,0.63727941,0.63973308,0.64224061,0.64480505,0.64742972,0.65011824,0.65287452,0.65570285,0.65860793,0.6615949,0.66466946,0.66783786,0.67110707,0.67448482,0.67797977,0.68160163,0.68536134,0.68927131,0.69334567,0.69760064,0.70205491,0.7067302,0.71165198,0.71685033,0.72236118,0.72822787,0.73450341,0.74125354,0.7485612,0.75653308,0.76530966,0.77508119,0.78611443,0.79880003,0.81374363,0.83195987,0.85535081,0.88818667,0.94409333,1.,1.,0.94409333,0.88818667,0.85535081,0.83195987,0.81374363,0.79880003,0.78611443,0.77508119,0.76530966,0.75653308,0.7485612,0.74125354,0.73450341,0.72822787,0.72236118,0.71685033,0.71165198,0.7067302,0.70205491,0.69760064,0.69334567,0.68927131,0.68536134,0.68160163,0.67797977,0.67448482,0.67110707,0.66783786,0.66466946,0.6615949,0.65860793,0.65570285,0.65287452,0.65011824,0.64742972,0.64480505,0.64224061,0.63973308,0.63727941,0.63487677,0.63252254,0.63021427,0.62794972,0.62572677,0.62354345,0.62139793,0.61928848,0.6172135,0.61517145,0.61316093,0.61118059,0.60922916,0.60730545,0.60540833,0.60353673,0.60168964,0.59986611,0.59806522,0.5962861,0.59452794,0.59278996,0.5910714,0.58937155,0.58768973,0.5860253,0.58437763,0.58274613,0.58113023,0.57952939,0.57794307,0.57637077,0.57481202,0.57326635,0.5717333,0.57021245,0.56870339,0.56720571,0.56571903,0.56424297,0.56277718,0.56132129,0.55987498,0.55843793,0.5570098,0.55559029,0.55417912,0.55277598,0.55138059,0.54999269,0.548612,0.54723826,0.54587123,0.54451066,0.5431563,0.54180793,0.54046531,0.53912822,0.53779645,0.53646977,0.53514798,0.53383087,0.53251824,0.53120989,0.52990563,0.52860527,0.52730862,0.5260155,0.52472572,0.5234391,0.52215548,0.52087466,0.51959649,0.51832079,0.51704739,0.51577613,0.51450685,0.51323937,0.51197354,0.5107092,0.50944618,0.50818433,0.5069235,0.50566352,0.50440424,0.5031455,0.50188714,0.50062902,0.49937098,0.49811286,0.4968545,0.49559576,0.49433648,0.4930765,0.49181567,0.49055382,0.4892908,0.48802646,0.48676063,0.48549315,0.48422387,0.48295261,0.48167921,0.48040351,0.47912534,0.47784452,0.4765609,0.47527428,0.4739845,0.47269138,0.47139473,0.47009437,0.46879011,0.46748176,0.46616913,0.46485202,0.46353023,0.46220355,0.46087178,0.45953469,0.45819207,0.4568437,0.45548934,0.45412877,0.45276174,0.451388,0.45000731,0.44861941,0.44722402,0.44582088,0.44440971,0.4429902,0.44156207,0.44012502,0.43867871,0.43722282,0.43575703,0.43428097,0.43279429,0.43129661,0.42978755,0.4282667,0.42673365,0.42518798,0.42362923,0.42205693,0.42047061,0.41886977,0.41725387,0.41562237,0.4139747,0.41231027,0.41062845,0.4089286,0.40721004,0.40547206,0.4037139,0.40193478,0.40013389,0.39831036,0.39646327,0.39459167,0.39269455,0.39077084,0.38881941,0.38683907,0.38482855,0.3827865,0.38071152,0.37860207,0.37645655,0.37427323,0.37205028,0.36978573,0.36747746,0.36512323,0.36272059,0.36026692,0.35775939,0.35519495,0.35257028,0.34988176,0.34712548,0.34429715,0.34139207,0.3384051,0.33533054,0.33216214,0.32889293,0.32551518,0.32202023,0.31839837,0.31463866,0.31072869,0.30665433,0.30239936,0.29794509,0.2932698,0.28834802,0.28314967,0.27763882,0.27177213,0.26549659,0.25874646,0.2514388,0.24346692,0.23469034,0.22491881,0.21388557,0.20119997,0.18625637,0.16804013,0.14464919,0.11181333,0.05590667,0.])
        self.synthesis = numpy.asarray([0.00000000e+00,2.52493737e-05,1.00993617e-04,2.27221124e-04,4.03912573e-04,6.31040943e-04,9.08571512e-04,1.23646188e-03,1.61466197e-03,2.04311406e-03,2.52175277e-03,3.05050510e-03,3.62929044e-03,4.25802059e-03,4.93659976e-03,5.66492464e-03,6.44288435e-03,7.27036053e-03,8.14722732e-03,9.07335139e-03,1.00485920e-02,1.10728009e-02,1.21458227e-02,1.32674943e-02,1.44376456e-02,1.56560990e-02,1.69226697e-02,1.82371657e-02,1.95993878e-02,2.10091294e-02,2.24661772e-02,2.39703105e-02,2.55213015e-02,2.71189155e-02,2.87629109e-02,3.04530390e-02,3.21890442e-02,3.39706641e-02,3.57976294e-02,3.76696640e-02,3.95864853e-02,4.15478036e-02,4.35533228e-02,4.56027403e-02,4.76957466e-02,4.98320260e-02,5.20112561e-02,5.42331081e-02,5.64972471e-02,5.88033315e-02,6.11510136e-02,6.35399396e-02,6.59697493e-02,6.84400765e-02,7.09505488e-02,7.35007880e-02,7.60904099e-02,7.87190242e-02,8.13862349e-02,8.40916401e-02,8.68348324e-02,8.96153985e-02,9.24329196e-02,9.52869711e-02,9.81771232e-02,1.01102941e-01,1.04063982e-01,1.07059803e-01,1.10089950e-01,1.13153968e-01,1.16251394e-01,1.19381763e-01,1.22544602e-01,1.25739435e-01,1.28965780e-01,1.32223151e-01,1.35511057e-01,1.38829001e-01,1.42176485e-01,1.45553002e-01,1.48958043e-01,1.52391095e-01,1.55851640e-01,1.59339154e-01,1.62853113e-01,1.66392985e-01,1.69958235e-01,1.73548325e-01,1.77162713e-01,1.80800852e-01,1.84462192e-01,1.88146180e-01,1.91852259e-01,1.95579867e-01,1.99328441e-01,2.03097413e-01,2.06886212e-01,2.10694266e-01,2.14520996e-01,2.18365823e-01,2.22228164e-01,2.26107433e-01,2.30003042e-01,2.33914400e-01,2.37840913e-01,2.41781985e-01,2.45737016e-01,2.49705407e-01,2.53686555e-01,2.57679853e-01,2.61684694e-01,2.65700470e-01,2.69726568e-01,2.73762377e-01,2.77807281e-01,2.81860664e-01,2.85921908e-01,2.89990394e-01,2.94065503e-01,2.98146611e-01,3.02233096e-01,3.06324335e-01,3.10419702e-01,3.14518572e-01,3.18620319e-01,3.22724316e-01,3.26829934e-01,3.30936547e-01,3.35043526e-01,3.39150246e-01,3.43256086e-01,3.47360427e-01,3.51462648e-01,3.55562129e-01,3.59658251e-01,3.63750398e-01,3.67837950e-01,3.71920292e-01,3.75996808e-01,3.80066883e-01,3.84129905e-01,3.88185260e-01,3.92232339e-01,3.96270531e-01,4.00299230e-01,4.04317828e-01,4.08325721e-01,4.12322306e-01,4.16306982e-01,4.20279150e-01,4.24238213e-01,4.28183575e-01,4.32114645e-01,4.36030831e-01,4.39931546e-01,4.43816203e-01,4.47684220e-01,4.51535015e-01,4.55368011e-01,4.59182632e-01,4.62978306e-01,4.66754464e-01,4.70510539e-01,4.74245967e-01,4.77960189e-01,4.81652647e-01,4.85322788e-01,4.88970060e-01,4.92593918e-01,4.96193817e-01,4.99769218e-01,5.03319584e-01,5.06844384e-01,5.10343088e-01,5.13815172e-01,5.17260116e-01,5.20677401e-01,5.24066516e-01,5.27426952e-01,5.30758205e-01,5.34059774e-01,5.37331165e-01,5.40571886e-01,5.43781450e-01,5.46959376e-01,5.50105185e-01,5.53218405e-01,5.56298569e-01,5.59345212e-01,5.62357878e-01,5.65336111e-01,5.68279465e-01,5.71187495e-01,5.74059764e-01,5.76895840e-01,5.79695293e-01,5.82457703e-01,5.85182652e-01,5.87869728e-01,5.90518527e-01,5.93128647e-01,5.95699693e-01,5.98231278e-01,6.00723016e-01,6.03174532e-01,6.05585452e-01,6.07955411e-01,6.10284050e-01,6.12571014e-01,6.14815956e-01,6.17018534e-01,6.19178413e-01,6.21295262e-01,6.23368760e-01,6.25398589e-01,6.27384439e-01,6.29326006e-01,6.31222993e-01,6.33075109e-01,6.34882068e-01,6.36643595e-01,6.38359416e-01,6.40029269e-01,6.41652895e-01,6.43230043e-01,6.44760469e-01,6.46243937e-01,6.47680215e-01,6.49069080e-01,6.50410317e-01,6.51703715e-01,6.52949072e-01,6.54146194e-01,6.55294893e-01,6.56394987e-01,6.57446303e-01,6.58448674e-01,6.59401943e-01,6.60305956e-01,6.61160570e-01,6.61965647e-01,6.62721059e-01,6.63426683e-01,6.64082404e-01,6.64688115e-01,6.65243717e-01,6.65749117e-01,6.66204231e-01,6.66608983e-01,6.66963302e-01,6.67267128e-01,6.67520406e-01,6.67723090e-01,6.67875141e-01,6.67976529e-01,6.68027230e-01,6.68027230e-01,6.67976529e-01,6.67875141e-01,6.67723090e-01,6.67520406e-01,6.67267128e-01,6.66963302e-01,6.66608983e-01,6.66204231e-01,6.65749117e-01,6.65243717e-01,6.64688115e-01,6.64082404e-01,6.63426683e-01,6.62721059e-01,6.61965647e-01,6.61160570e-01,6.60305956e-01,6.59401943e-01,6.58448674e-01,6.57446303e-01,6.56394987e-01,6.55294893e-01,6.54146194e-01,6.52949072e-01,6.51703715e-01,6.50410317e-01,6.49069080e-01,6.47680215e-01,6.46243937e-01,6.44760469e-01,6.43230043e-01,6.41652895e-01,6.40029269e-01,6.38359416e-01,6.36643595e-01,6.34882068e-01,6.33075109e-01,6.31222993e-01,6.29326006e-01,6.27384439e-01,6.25398589e-01,6.23368760e-01,6.21295262e-01,6.19178413e-01,6.17018534e-01,6.14815956e-01,6.12571014e-01,6.10284050e-01,6.07955411e-01,6.05585452e-01,6.03174532e-01,6.00723016e-01,5.98231278e-01,5.95699693e-01,5.93128647e-01,5.90518527e-01,5.87869728e-01,5.85182652e-01,5.82457703e-01,5.79695293e-01,5.76895840e-01,5.74059764e-01,5.71187495e-01,5.68279465e-01,5.65336111e-01,5.62357878e-01,5.59345212e-01,5.56298569e-01,5.53218405e-01,5.50105185e-01,5.46959376e-01,5.43781450e-01,5.40571886e-01,5.37331165e-01,5.34059774e-01,5.30758205e-01,5.27426952e-01,5.24066516e-01,5.20677401e-01,5.17260116e-01,5.13815172e-01,5.10343088e-01,5.06844384e-01,5.03319584e-01,4.99769218e-01,4.96193817e-01,4.92593918e-01,4.88970060e-01,4.85322788e-01,4.81652647e-01,4.77960189e-01,4.74245967e-01,4.70510539e-01,4.66754464e-01,4.62978306e-01,4.59182632e-01,4.55368011e-01,4.51535015e-01,4.47684220e-01,4.43816203e-01,4.39931546e-01,4.36030831e-01,4.32114645e-01,4.28183575e-01,4.24238213e-01,4.20279150e-01,4.16306982e-01,4.12322306e-01,4.08325721e-01,4.04317828e-01,4.00299230e-01,3.96270531e-01,3.92232339e-01,3.88185260e-01,3.84129905e-01,3.80066883e-01,3.75996808e-01,3.71920292e-01,3.67837950e-01,3.63750398e-01,3.59658251e-01,3.55562129e-01,3.51462648e-01,3.47360427e-01,3.43256086e-01,3.39150246e-01,3.35043526e-01,3.30936547e-01,3.26829934e-01,3.22724316e-01,3.18620319e-01,3.14518572e-01,3.10419702e-01,3.06324335e-01,3.02233096e-01,2.98146611e-01,2.94065503e-01,2.89990394e-01,2.85921908e-01,2.81860664e-01,2.77807281e-01,2.73762377e-01,2.69726568e-01,2.65700470e-01,2.61684694e-01,2.57679853e-01,2.53686555e-01,2.49705407e-01,2.45737016e-01,2.41781985e-01,2.37840913e-01,2.33914400e-01,2.30003042e-01,2.26107433e-01,2.22228164e-01,2.18365823e-01,2.14520996e-01,2.10694266e-01,2.06886212e-01,2.03097413e-01,1.99328441e-01,1.95579867e-01,1.91852259e-01,1.88146180e-01,1.84462192e-01,1.80800852e-01,1.77162713e-01,1.73548325e-01,1.69958235e-01,1.66392985e-01,1.62853113e-01,1.59339154e-01,1.55851640e-01,1.52391095e-01,1.48958043e-01,1.45553002e-01,1.42176485e-01,1.38829001e-01,1.35511057e-01,1.32223151e-01,1.28965780e-01,1.25739435e-01,1.22544602e-01,1.19381763e-01,1.16251394e-01,1.13153968e-01,1.10089950e-01,1.07059803e-01,1.04063982e-01,1.01102941e-01,9.81771232e-02,9.52869711e-02,9.24329196e-02,8.96153985e-02,8.68348324e-02,8.40916401e-02,8.13862349e-02,7.87190242e-02,7.60904099e-02,7.35007880e-02,7.09505488e-02,6.84400765e-02,6.59697493e-02,6.35399396e-02,6.11510136e-02,5.88033315e-02,5.64972471e-02,5.42331081e-02,5.20112561e-02,4.98320260e-02,4.76957466e-02,4.56027403e-02,4.35533228e-02,4.15478036e-02,3.95864853e-02,3.76696640e-02,3.57976294e-02,3.39706641e-02,3.21890442e-02,3.04530390e-02,2.87629109e-02,2.71189155e-02,2.55213015e-02,2.39703105e-02,2.24661772e-02,2.10091294e-02,1.95993878e-02,1.82371657e-02,1.69226697e-02,1.56560990e-02,1.44376456e-02,1.32674943e-02,1.21458227e-02,1.10728009e-02,1.00485920e-02,9.07335139e-03,8.14722732e-03,7.27036053e-03,6.44288435e-03,5.66492464e-03,4.93659976e-03,4.25802059e-03,3.62929044e-03,3.05050510e-03,2.52175277e-03,2.04311406e-03,1.61466197e-03,1.23646188e-03,9.08571512e-04,6.31040943e-04,4.03912573e-04,2.27221124e-04,1.00993617e-04,2.52493737e-05,0.00000000e+00])
        self.stft = STFT(512, hop=self.hop, analysis_window=self.hann,synthesis_window=self.synthesis ,online=True)
        self.logit = numpy.zeros(((self.NBINS,192)),dtype=numpy.complex128,order='C')
        self.residue = numpy.zeros(((self.NBINS,192)),dtype=numpy.complex128,order='C')
        self.logit_real = numpy.zeros(((self.NBINS,192)),dtype=numpy.float64,order='C')

        self.result = numpy.zeros(((257,64)),dtype=numpy.complex128,order='C')
        self.zeros = numpy.zeros(((64,257)),dtype=numpy.complex128,order='C')
        self.mask = numpy.zeros(((257,192)),dtype=numpy.float64,order='C')
        self.previous = numpy.zeros(((257,192)),dtype=numpy.float64,order='C')
        self.audio = numpy.zeros(8192*3,dtype=numpy.float32,order='C')
        self.constant = 0.057

    def process(self,data):

      #24000/256 = 93.75 hz per frequency bin.
      #a 4000 hz window(the largest for SSB is roughly 43 bins.
      #https://en.wikipedia.org/wiki/Voice_frequency
      #however, practically speaking, voice frequency cuts off just above 3400hz on SSB.
      #so we use 36 bins in total and catch most speech activity below 3400hz.
      #we could do a better job of achieving selectivity if we allocated more bins to frequency and less to time,
      #but all the other settings we could consider do not provide so general and easy a framework as 512 and 128.
      #resampling everything to 48000, or using hop/4 for 24000,12000,6000, and resampling to closest delivers nearly identical results.
      #given that 48000 is a standard used almost everywhere, this works out optimally.


      self.audio[:-8192] = self.audio[8192:] #numpy.roll(self.audio,-8192)
      self.audio[-8192:] = data[:]
      self.logit[:] = sstft(self.audio,self.logistic,512,hop_len=128,dtype= 'float64')[0:self.NBINS,:]
      self.logit_real[:] = numpy.abs(self.logit)


      entropy_unmasked = fast_entropy(numpy.sort(self.logit_real,axis=0))
      entropy_unmasked[numpy.isnan(entropy_unmasked)] = 0
      entropy = smoothpadded(entropy_unmasked,3).astype(dtype=numpy.float64)

      if numpy.max(entropy) > self.constant: 
        entropy[entropy<self.constant] = 0
        entropy[entropy>0] = 1
        entropy = entropy.astype(dtype=numpy.int64)

        count = numpy.sum(entropy[64-32:128+32])
        maxstreak = longestConsecutive(entropy[64-32:128+32])
        if count>22 or maxstreak>16:

            entropy[:]= remove_outliers(entropy,0,6,1)
            entropy[:]= remove_outliers(entropy,1,2,0)
            t = threshold(numpy.ravel(self.logit_real))
            self.logit_real[:] = sawtooth_filter(self.logit_real[:])
            self.mask.fill(0)
            self.previous.fill(0)

            self.previous[0:self.NBINS,:] = fast_peaks(self.logit_real,entropy,t,entropy_unmasked)

            self.residue[:] =  self.logit - (self.logit *  self.previous[0:self.NBINS,:])
            previous = self.logit_real.max()
            self.logit_real[:] = abs(self.residue[:])
            multiplier = self.logit_real.max()/previous
            self.logit_real[:] = sawtooth_filter(self.logit_real[:])

            self.mask[0:self.NBINS,:] = fast_peaks(self.logit_real,entropy,t,entropy_unmasked)     

            self.mask[:] = numpy.maximum(self.mask*multiplier,self.previous)

            self.mask[:] = sawtooth_filter(self.mask)
            self.mask[:] = convolve_custom_filter_2d(self.mask[:],13,3,3)

            #then we do it again on the remainder to maximally extract the desired waveform
            #we retain our entropy encoding
            self.result[:] = sstft(self.audio,self.hann,512,hop_len=128,dtype= 'float64')[:,64:128] * self.mask[:,64:128]
            return self.stft.synthesis(self.result.T)
      return self.stft.synthesis(self.zeros)

        
def padarray(A, size):
    t = size - len(A)
    return numpy.pad(A, pad_width=(0, t), mode='constant',constant_values=numpy.std(A))

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

import time
def process_data(data: numpy.ndarray):
    print("processing ", data.size / rate, " seconds long file at ", rate, " rate.")
    start = time.time()
    filter = Filter()
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
        self.rightfilter = Filter()
        self.leftfilter = Filter()
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
        chans.append(self.rightfilter.process(audio_in[:,0]))
        chans.append(self.leftfilter.process(audio_in[:,1]))

        self.speakerstream.write(numpy.column_stack(chans).astype(numpy.float32).tobytes())
        return None, pyaudio.paContinue


    def stream_start(self):
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
