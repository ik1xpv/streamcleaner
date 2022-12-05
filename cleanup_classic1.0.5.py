#idea and code and bugs by Joshuah Rainstar   : https://groups.io/g/NextGenSDRs/message/1085
#fork, mod by Oscar Steila : https://groups.io/g/NextGenSDRs/topic/spectral_denoising
#cleanup_classic1.0.5.py

#12/5/2022 : Warning. This is only an experiment and not for the faint of heart
#For proper denoising, you can use denoising tools built into your SDR, or use VST plugins.
#One excellent plugin is acon digital denoise 2. It is realtime. But it does cost $99.
#This experiment is intended to explore improvements to common threshholding and denoising methods in a domain
#which is readily accessible for experimentation, namely simple scripting without any complex languages or compilers.
#Python's performance has come a long way and with numpy, numba, and a little optimization, it is closer to c++.
#Therefore consider this merely a testbed. Feel free to try your own algorithms and optimizations in here.
#At the moment, this testbed is hosting an improvised thresholding algorithm.
#This is inspired by the noisereduce python library, which works somewhat well but was designed for bird calls.
#The algorithm assumes that the signal is above the noise floor, attempts to estimate the noise floor, and then
#calculates a threshold high enough to identify the voice formant waveform ridges, which is then softened to form
#a "contour" around the signal. It also works well on CW and some data modes, but is only intended for listening.
#Science has concluded the harsher the noise, the more it stresses the listener. It can be harmful to health.
#We recommend using this in a pipeline with an expander(agc) after the noise reduction.
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
#pip install librosa pipwin numba dearpygui np-rw-buffer opencv-python
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
#I recommend adding 3db of ephram malah in your SDR(or speex, which is enhanced ephram-malah, if any of them have it)
#and 10db of gain after this script, followed by an expander and a compressor- if you know how to use one.

import os
import numpy
import numpy as np

import numba
import pyaudio
import librosa
import warnings
import scipy

from time import sleep
from np_rw_buffer import AudioFramingBuffer
import dearpygui.dearpygui as dpg

from scipy.signal._arraytools import const_ext, even_ext, odd_ext, zero_ext
from scipy.signal.windows import get_window
from scipy.signal.signaltools import detrend
from scipy import interpolate as interp
from scipy.special import logit
from scipy import fftpack


def padarray(A, size):
    t = size - len(A)
    return numpy.pad(A, pad_width=(0, t), mode='constant',constant_values=numpy.std(A))

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def _triage_segments(window, nperseg, input_length):
    """
    Parses window and nperseg arguments for spectrogram and _spectral_helper.
    This is a helper function, not meant to be called externally.
    Parameters
    ----------
    window : string, tuple, or ndarray
        If window is specified by a string or tuple and nperseg is not
        specified, nperseg is set to the default of 256 and returns a window of
        that length.
        If instead the window is array_like and nperseg is not specified, then
        nperseg is set to the length of the window. A ValueError is raised if
        the user supplies both an array_like window and a value for nperseg but
        nperseg does not equal the length of the window.
    nperseg : int
        Length of each segment
    input_length: int
        Length of input signal, i.e. x.shape[-1]. Used to test for errors.
    Returns
    -------
    win : ndarray
        window. If function was called with string or tuple than this will hold
        the actual array used as a window.
    nperseg : int
        Length of each segment. If window is str or tuple, nperseg is set to
        256. If window is array_like, nperseg is set to the length of the
        window.
    """
    # parse window; if array like, then set nperseg = win.shape
    if isinstance(window, str) or isinstance(window, tuple):
        # if nperseg not specified
        if nperseg is None:
            nperseg = 256  # then change to default
        if nperseg > input_length:
            warnings.warn('nperseg = {0:d} is greater than input length '
                          ' = {1:d}, using nperseg = {1:d}'
                          .format(nperseg, input_length))
            nperseg = input_length
        win = get_window(window, nperseg)
    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if input_length < win.shape[-1]:
            raise ValueError('window is longer than input signal')
        if nperseg is None:
            nperseg = win.shape[0]
        elif nperseg is not None:
            if nperseg != win.shape[0]:
                raise ValueError("value specified for nperseg is different"
                                 " from length of window")
    return win, nperseg

def spectral_helper(x, fs=1.0, window='hann', nperseg=None, noverlap=None,
                     nfft=None, detrend='constant', return_onesided=True,
                     scaling='density', axis=-1, mode='psd', boundary=None,
                     padded=False):

    if mode not in ['psd', 'stft']:
        raise ValueError("Unknown value for mode %s, must be one of: "
                         "{'psd', 'stft'}" % mode)

    boundary_funcs = {'even': even_ext,
                      'odd': odd_ext,
                      'constant': const_ext,
                      'zeros': zero_ext,
                      None: None}

    if boundary not in boundary_funcs:
        raise ValueError("Unknown boundary option '{0}', must be one of: {1}"
                         .format(boundary, list(boundary_funcs.keys())))

    # If x and y are the same object we can save ourselves some computation.
    same_data = True
    y = []

    if not same_data and mode != 'psd':
        raise ValueError("x and y must be equal if mode is 'stft'")

    axis = int(axis)

    # Ensure we have np.arrays, get outdtype
    x = np.asarray(x)
    if not same_data:
        y = np.asarray(y)
        outdtype = np.result_type(x, y, np.complex64)
    else:
        outdtype = np.result_type(x, np.complex64)

    if not same_data:
        # Check if we can broadcast the outer axes together
        xouter = list(x.shape)
        youter = list(y.shape)
        xouter.pop(axis)
        youter.pop(axis)
        try:
            outershape = np.broadcast(np.empty(xouter), np.empty(youter)).shape
        except ValueError as e:
            raise ValueError('x and y cannot be broadcast together.') from e

    if same_data:
        if x.size == 0:
            return np.empty(x.shape), np.empty(x.shape), np.empty(x.shape)
    else:
        if x.size == 0 or y.size == 0:
            outshape = outershape + (min([x.shape[axis], y.shape[axis]]),)
            emptyout = np.moveaxis(np.empty(outshape), -1, axis)
            return emptyout, emptyout, emptyout

    if x.ndim > 1:
        if axis != -1:
            x = np.moveaxis(x, axis, -1)
            if not same_data and y.ndim > 1:
                y = np.moveaxis(y, axis, -1)

    # Check if x and y are the same length, zero-pad if necessary
    if not same_data:
        if x.shape[-1] != y.shape[-1]:
            if x.shape[-1] < y.shape[-1]:
                pad_shape = list(x.shape)
                pad_shape[-1] = y.shape[-1] - x.shape[-1]
                x = np.concatenate((x, np.zeros(pad_shape)), -1)
            else:
                pad_shape = list(y.shape)
                pad_shape[-1] = x.shape[-1] - y.shape[-1]
                y = np.concatenate((y, np.zeros(pad_shape)), -1)

    if nperseg is not None:  # if specified by user
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')

    # parse window; if array like, then set nperseg = win.shape
    win, nperseg = _triage_segments(window, nperseg, input_length=x.shape[-1])

    if nfft is None:
        nfft = nperseg
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)

    if noverlap is None:
        noverlap = nperseg//2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nperseg - noverlap

    # Padding occurs after boundary extension, so that the extended signal ends
    # in zeros, instead of introducing an impulse at the end.
    # I.e. if x = [..., 3, 2]
    # extend then pad -> [..., 3, 2, 2, 3, 0, 0, 0]
    # pad then extend -> [..., 3, 2, 0, 0, 0, 2, 3]

    if boundary is not None:
        ext_func = boundary_funcs[boundary]
        x = ext_func(x, nperseg//2, axis=-1)
        if not same_data:
            y = ext_func(y, nperseg//2, axis=-1)

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = nperseg + (nseg-1)*nstep, with integer nseg
        nadd = (-(x.shape[-1]-nperseg) % nstep) % nperseg
        zeros_shape = list(x.shape[:-1]) + [nadd]
        x = np.concatenate((x, np.zeros(zeros_shape)), axis=-1)
        if not same_data:
            zeros_shape = list(y.shape[:-1]) + [nadd]
            y = np.concatenate((y, np.zeros(zeros_shape)), axis=-1)

    # Handle detrending and window functions
    if not detrend:
        def detrend_func(d):
            return d
    elif not hasattr(detrend, '__call__'):
        def detrend_func(d):
            return detrend(d, type=detrend, axis=-1)
    elif axis != -1:
        # Wrap this function so that it receives a shape that it could
        # reasonably expect to receive.
        def detrend_func(d):
            d = np.moveaxis(d, -1, axis)
            d = detrend(d)
            return np.moveaxis(d, axis, -1)
    else:
        detrend_func = detrend

    if np.result_type(win, np.complex64) != outdtype:
        win = win.astype(outdtype)

    if scaling == 'density':
        scale = 1.0 / (fs * (win*win).sum())
    elif scaling == 'spectrum':
        scale = 1.0 / win.sum()**2
    else:
        raise ValueError('Unknown scaling: %r' % scaling)

    if mode == 'stft':
        scale = np.sqrt(scale)

    if return_onesided:
        if np.iscomplexobj(x):
            sides = 'twosided'
            warnings.warn('Input data is complex, switching to '
                          'return_onesided=False')
        else:
            sides = 'onesided'
            if not same_data:
                if np.iscomplexobj(y):
                    sides = 'twosided'
                    warnings.warn('Input data is complex, switching to '
                                  'return_onesided=False')

    # Perform the windowed FFTs
    result = _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides)

   

    result *= scale
    if sides == 'onesided' and mode == 'psd':
        if nfft % 2:
            result[..., 1:] *= 2
        else:
            # Last point is unpaired Nyquist freq point, don't double
            result[..., 1:-1] *= 2

    result = result.astype(outdtype)

    # All imaginary parts are zero anyways
    if same_data and mode != 'stft':
        result = result.real

    # Output is going to have new last axis for time/window index, so a
    # negative axis index shifts down one
    if axis < 0:
        axis -= 1

    # Roll frequency axis back to axis where the data came from
    result = np.moveaxis(result, -1, axis)

    return result


def _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides):

    # Created strided array of data segments
    if nperseg == 1 and noverlap == 0:
        result = x[..., numpy.newaxis]
    else:
        # https://stackoverflow.com/a/5568169
        step = nperseg - noverlap
        shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
        strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                                 strides=strides)

    # Detrend each data segment individually
    return result

def fast_hamming(M, sym=True):
    a = [0.5, 0.5]
    fac = numpy.linspace(-numpy.pi, numpy.pi, M)
    w = numpy.zeros(M)
    for k in range(len(a)):
        w += a[k] * numpy.cos(k * fac)
    return w

def broken_hamming(M, sym=True):
    a = [0.5, 0.5]
    fac = numpy.linspace(-numpy.pi, numpy.pi, M)
    w = numpy.zeros(M)
    for k in range(len(a)):
        w += a[k] * numpy.cos(k * fac)
        return w

def cosine_bell(M, sym=True):
    a = [0.5, 0.5]
    fac = numpy.linspace(-numpy.pi, numpy.pi, M)
    w = numpy.zeros(M)
    for k in range(len(a)):
        w += a[k] * numpy.cos(k * fac)
    return w

def man(arr):

    med = numpy.nanmedian(arr[numpy.nonzero(arr)])
    return numpy.nanmedian(numpy.abs(arr - med))

def atd(arr):
    x = numpy.square(abs(arr - man(arr)))
    return numpy.sqrt(numpy.nanmean(x))

def aSNR(arr):
  return man(arr)/atd(arr)

def threshhold(arr):
  return (atd(arr)+ numpy.nanmedian(arr[numpy.nonzero(arr)])) 


#https://sci-hub.se/10.1109/taslp.2017.2747082
#New Results in Modulation-Domain Single-Channel Speech Enhancement
#IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 25, NO. 11, NOVEMBER 2017 2125
#Pejman Mowlaee, Senior Member, IEEE, Martin Blass, Student Member, IEEE, and W. Bastiaan Kleijn, Fellow, IEEE
#instantaneous pitch period P0 = 600 = 48000 / 80(fundemental pitch chosen)
#Let L be the number of time frames within one time block, each of length P0
#The MLT is implemented using the DCT-IV in combination with a square-root Hann window of length 2P˜0 and 50% overlap [25].



def entropy(data: numpy.ndarray):
    a = data.copy()
    a = numpy.sort(a)
    scaled = numpy.interp(a, (a.min(), a.max()), (-6, +6))
    fprint = numpy.linspace(0, 1, a.size)
    y = logit(fprint)
    y[y == -numpy.inf] = -6
    y[y == +numpy.inf] = 6
    z = numpy.corrcoef(scaled, y)
    completeness = z[0, 1]
    sigma = 1 - completeness
    return sigma

def ds(data: numpy.ndarray):
  data = data.copy() #i dont know why this is needed
  frames = []

  for each in chunks(data, 1200):
      if each.size == 1200:
          d_working = spectral_helper(each,window=numpy.sqrt(cosine_bell(1200)),mode='psd',nperseg=1200,noverlap=600,detrend=None).T
          d_working = numpy.flip(d_working,axis=1)

          mlt = scipy.fftpack.dct(d_working,type=4)


          frames.append(mlt)
      else:
        each = padarray(each, 1200)
        d_working = spectral_helper(each,window=numpy.sqrt(cosine_bell(1200)),mode='psd',nperseg=1200,noverlap=600,detrend=None).T
        d_working = numpy.flip(d_working,axis=1)
        mlt = scipy.fftpack.dct(d_working,type=4)


        frames.append(mlt)

   #L = 0, 1,...,L − 1 # frame index
   #k = 0, 1,..., P˜0 − 1 # frequency channel index, respectively [5].
   #The first transform outputs the MLT coefficients f (l, k) which slowly evolve over time for a periodic
   #signal segment but rapidly changes for an aperiodic segment.


   #todo: apply the pitch shifting transform/interpolation.

   #Modulation Transform: By applying a pitch-synchronous
   #transform on speech signal we obtain L consecutive MLT coefficient frames.  = nframes
   #These MLT coefficients are then merged into one segment.
  nframes = len(frames) #get the number of frames
  frames = numpy.asarray(frames)
  frames = numpy.squeeze(frames)

   
   # A DCT-II is applied to perform the modulation transform on this segment for each frequency channel
   #k = 0, ··· , K − 1. The choice of DCT-II together with a rectangular window facilitates the implementation of the modulation
   #transform as a critically sampled filter [25] which takes into
   #account the rapid onsets of speech
  #nperseg = nsamples 
  #The outcome of the TBS is a constant number of time frames L per time block (of length P˜0) and thus a fixed number of modulation bands Q per DS.
  #therefore
  #l = nframes * 2
  #As a trade-off between male and female speakers, we choose Q = L = 4, used in our experiments.


  #at 48000 samples:
  #a time block here of p0 is 600, which is transformed in 1200 chunks and becomes 40 frequency channel indexes and 1200 samples in one second.
  # Now do we just use k? no, we have to use L. L, coincidentally, for this format, is 160(twice the number of time frames) because each frame isn't p0  but 2*p0
  second_frames = []
  l = nframes * 2

  for each in range(nframes):
      
          d_working = spectral_helper(frames[each,:],window=cosine_bell(l),mode='psd',nperseg=l,noverlap=l//2,detrend=None)
          mlt = scipy.fftpack.dct(d_working,type=2)
          second_frames.append(mlt)
  frames = numpy.asarray(second_frames)
  frames = numpy.rot90(frames)
  frames = frames.reshape(l,-1)
  frames = numpy.flip(frames,axis=0)

  result_r = numpy.square(frames.real) + numpy.square(frames.imag) #obtain absolute**2
  Y = result_r[~numpy.isnan(result_r)]
  max = numpy.where(numpy.isinf(Y),0,Y).argmax()
  max = Y[max]
  result_r = numpy.nan_to_num(result_r, copy=True, nan=0.0, posinf=max, neginf=0.0)#correct irrationalities
  result_r = numpy.sqrt(result_r) #stft_vr >= 0 
  result_r=(result_r-np.nanmin(result_r))/np.ptp(result_r)

  return result_r

#it's possible im doing this wrong and this could be done faster with a native numpy function.
#all we do here is take the 2d image and the transpose of the image, and unravel them both.
#given that the ravel in this case doesn't edit the contents(ie a different formant or transpose AFTER the ravel),
#the ravel is just a view- any change to the flattened form happens to the primary form.
#We then smooth both, and then transpose back the transposed copy, and recombine them.
#as a result we have our 2d smoothing with nearest neighbor. 
#@numba.jit(numba.float64[:,:](numba.float64[:,:]),cache=True,parallel=True,nogil=True)
#replacing the decorator with the above will speed up the code, but its possible to make changes(to this function body)
#that could then cause improper memory access/writing and crash python, so I don't use it by default, but you can swap it
#for an immediate(perhaps dramatic) speed boost.


@numba.jit(numba.float64[:,:](numba.float64[:,:]),cache=True,parallel=True,nogil=True)
def numba_adjacent_filter(data: numpy.ndarray):
        normal = data.copy()
        transposed = data.copy()
        transposed = transposed.T
        transposed_raveled = numpy.ravel(transposed)
        normal_raveled = numpy.ravel(normal)
        zeroth = numpy.zeros_like(normal_raveled)
        
        for i in numba.prange(1,zeroth.size - 1):
            zeroth[i] = (transposed_raveled[i - 1] + transposed_raveled[i] + transposed_raveled[i + 1]) / 3
        
        zeroth[0] = (transposed_raveled[0] + (transposed_raveled[1] + transposed_raveled[2]) / 2) / 3
        zeroth[-1] = (transposed_raveled[-1] + (transposed_raveled[-2] + transposed_raveled[-3]) / 2) / 3
        transposed_raveled[:] = zeroth.copy()
        transposed = transposed.T

        for i in numba.prange(1,zeroth.size - 1):
            zeroth[i] = (normal_raveled[i - 1] + normal_raveled[i] + normal_raveled[i + 1]) / 3
        
        zeroth[0] = (normal_raveled[0] + (normal_raveled[1] + normal_raveled[2]) / 2) / 3
        zeroth[-1] = (normal_raveled[-1] + (normal_raveled[-2] + normal_raveled[-3]) / 2) / 3
        normal_raveled[:] = zeroth.copy()

        return (transposed + normal)/2

@numba.jit(numba.float64[:,:](numba.float64[:,:]),cache=True)
def filter_wrapper_50(data: numpy.ndarray):
  d = data.copy()
  for i in range(50):
    d = numba_adjacent_filter(d)
  return d

@numba.njit(cache=True)
def loop(h, w, k, kernel, x_pad, out):
    for row in range(h):
      for col in range(w):
        csum = 0.0
        for i in range(-k, k+1):
          csum += kernel[i+k] * x_pad[row+k, col+i+k]
        out[row,col] = csum
    return out


def filter(x):
    """Filter the image.
    """
    #sigma = 4.0
    #var = sigma * sigma
    k = 12 #int(3.0 * 4 + 0.5)
    q = numpy.linspace(-k, k, 2*k+1)
    kernel = numpy.exp(-0.5 * q * q / 16) #var)
    kernel = kernel / numpy.sum(kernel)
    h, w = x.shape
    x_pad = numpy.pad(x[:, :], [(k,), (k,)], mode="edge")
    horiz = numpy.empty_like(x, dtype=float)
    horiz = loop(h, w, k, kernel, x_pad, horiz)
    horiz = horiz.transpose((1, 0))
    h, w = horiz.shape
    x_pad = numpy.pad(horiz[:, :], [(k,), (k,)], mode="edge")
    horiz = loop(h, w, k, kernel, x_pad, horiz)
    horiz = horiz.transpose((1, 0)).astype(float)
    return horiz

def denoise(data: numpy.ndarray):
    data= numpy.asarray(data,dtype=float) #correct byte order of array

    #doublespectrum  = ds(data) #for later, this function is not optimized enough for realtime
    


    stft_r = librosa.stft(data,n_fft=512,window=fast_hamming) #get complex representation
    stft_vr = numpy.square(stft_r.real) + numpy.square(stft_r.imag) #obtain absolute**2
    Y = stft_vr[~numpy.isnan(stft_vr)]
    max = numpy.where(numpy.isinf(Y),-numpy.Inf,Y).argmax()
    max = Y[max]
    stft_vr = numpy.nan_to_num(stft_vr, copy=True, nan=0.0, posinf=max, neginf=0.0)#correct irrationalities
    stft_vr = numpy.sqrt(stft_vr) #stft_vr >= 0 
    stft_vr=(stft_vr-numpy.nanmin(stft_vr))/numpy.ptp(stft_vr) #normalize to 0,1
    ent = numpy.apply_along_axis(func1d=entropy,axis=0,arr=stft_vr)
    ent=(ent-numpy.nanmin(ent))/numpy.ptp(ent)#correct basis    

    t = threshhold(stft_vr)     
    mask_one = numpy.where(stft_vr>=t, 1,0)
    stft_demo = numpy.where(mask_one == 0, stft_vr,0)
    stft_d = stft_demo.flatten()
    stft_d = stft_d[stft_d>0]
    r = man(stft_d) #obtain a noise background basis



    t = threshhold(stft_vr[stft_vr>=t])   #obtain the halfway threshold
    mask_two = numpy.where(stft_vr>=t/2, 1.0,0)
    ent[ent<0.5] = 0.5


    mask = mask_two * ent[None,:] #reduce noise using an entropy filter
    mask[mask==0] = r #reduce warbling, you could also try r/2 or r/10 or something like that, its not as important

    mask = filter_wrapper_50(mask)
    #we now have two filters, and we should select criteria among them
    mask=(mask-numpy.nanmin(mask))/numpy.ptp(mask)#correct basis    

     
    stft_r = stft_r * mask

    processed = librosa.istft(stft_r,window=fast_hamming)
    return processed

import time
def process_data(data: numpy.ndarray):
    print("processing ", data.size / rate, " seconds long file at ", rate, " rate.")
    start = time.time()
    processed = []
    for each in chunks(data, rate):
        if each.size == rate:
            processed.append(denoise(each))
        else:
            psize = each.size
            working = padarray(each, rate)
            working = denoise(working)
            processed.append(working[0:psize])
    end = time.time()
    print("took ", end - start, " to process ", data.size / rate)
    return numpy.concatenate((processed), axis=0)   


class StreamSampler(object):
    dtype_to_paformat = {
        # Numpy dtype : pyaudio enum
        'uint8': pyaudio.paUInt8,
        'int8': pyaudio.paInt8,
        'uint16': pyaudio.paInt16,
        'int16': pyaudio.paInt16,
        'uint24': pyaudio.paInt24,
        'int24': pyaudio.paInt24,
        "uint32": pyaudio.paInt32,
        'int32': pyaudio.paInt32,
        'float32': pyaudio.paFloat32,

        # Float64 is not a valid pyaudio type.
        # The encode method changes this to a float32 before sending to audio
        'float64': pyaudio.paFloat32,
        "complex128": pyaudio.paFloat32,
    }

    @classmethod
    def get_pa_format(cls, dtype):
        try:
            dtype = dtype.dtype
        except (AttributeError, Exception):
            pass
        return cls.dtype_to_paformat[dtype.name]

    def __init__(self, sample_rate=48000, channels=2, buffer_delay=2,  # or 1.5, measured in seconds
                 micindex=1, speakerindex=1, dtype=numpy.float32):
        self.pa = pyaudio.PyAudio()
        self._processing_size = sample_rate
        # np_rw_buffer (AudioFramingBuffer offers a delay time)
        self._sample_rate = sample_rate
        self._channels = channels
        self.ticker = 0
        self.rb = AudioFramingBuffer(sample_rate=sample_rate, channels=channels,
                                     seconds=1,  # Buffer size (need larger than processing size)[seconds * sample_rate]
                                     buffer_delay=0,  # #this buffer doesnt need to have a size
                                     dtype=numpy.dtype(dtype))

        self.micindex = micindex
        self.speakerindex = speakerindex
        self.micstream = None
        self.speakerstream = None
        self.speakerdevice = ""
        self.micdevice = ""

        # Set inputs for inheritance
        self.set_sample_rate(sample_rate)
        self.set_channels(channels)
        self.set_dtype(dtype)

    @property
    def processing_size(self):
        return self._processing_size

    @processing_size.setter
    def processing_size(self, value):
        self._processing_size = value
        self._update_streams()	

    def get_sample_rate(self):
        return self._sample_rate

    def set_sample_rate(self, value):
        self._sample_rate = value
        try:  # RingBuffer
            self.rb.maxsize = int(value * 5)
        except AttributeError:
            pass
        try:  # AudioFramingBuffer
            self.rb.sample_rate = value
        except AttributeError:
            pass
        self._update_streams()

    sample_rate = property(get_sample_rate, set_sample_rate)

    def get_channels(self):
        return self._channels

    def set_channels(self, value):
        self._channels = value
        try:  # RingBuffer
            self.rb.columns = value
        except AttributeError:
            pass
        try:  # AudioFrammingBuffer
            self.rb.channels = value
        except AttributeError:
            pass
        self._update_streams()

    channels = property(get_channels, set_channels)

    def get_dtype(self):
        return self.rb.dtype

    def set_dtype(self, value):
        try:
            self.rb.dtype = value
        except AttributeError:
            pass
        self._update_streams()

    dtype = property(get_dtype, set_dtype)

    @property
    def pa_format(self):
        return self.get_pa_format(self.dtype)

    @pa_format.setter
    def pa_format(self, value):
        for np_dtype, pa_fmt in self.dtype_to_paformat.items():
            if value == pa_fmt:
                self.dtype = numpy.dtype(np_dtype)
                return

        raise ValueError('Invalid pyaudio format given!')

    @property
    def buffer_delay(self):
        try:
            return self.rb.buffer_delay
        except (AttributeError, Exception):
            return 0

    @buffer_delay.setter
    def buffer_delay(self, value):
        try:
            self.rb.buffer_delay = value
        except AttributeError:
            pass

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
            self.filterthread.join()
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

        stream = self.pa.open(format=self.pa_format,
                              channels=self.channels,
                              rate=int(self.sample_rate),
                              input=True,
                              input_device_index=self.micindex,  # device_index,
                              # each frame carries twice the data of the frames
                              frames_per_buffer=int(self._processing_size),
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

        stream = self.pa.open(format=self.pa_format,
                              channels=self.channels,
                              rate=int(self.sample_rate),
                              output=True,
                              output_device_index=self.speakerindex,
                              frames_per_buffer=int(self._processing_size),
                              stream_callback=self.non_blocking_stream_write,
                              start=False  # Need start to be False if you don't want this to start right away
                              )
        return stream

    # it is critical that this function do as little as possible, as fast as possible. numpy.ndarray is the fastest we can move.
    # attention: numpy.ndarray is actually faster than frombuffer for known buffer sizes
    def non_blocking_stream_read(self, in_data, frame_count, time_info, status):
        audio_in = numpy.ndarray(buffer=in_data, dtype=self.dtype,
                                            shape=[int(self._processing_size * self._channels)]).reshape(-1,
                                                                                                         self.channels)
        self.rb.write(audio_in, error=False)
        return None, pyaudio.paContinue

    def non_blocking_stream_write(self, in_data, frame_count, time_info, status):
        # Read raw data            
        if len(self.rb) < self._processing_size:
            print('Not enough data to play! Increase the buffer_delay')
            # uncomment this for debug
            audio = numpy.zeros((self._processing_size, self.channels), dtype=self.dtype)
            return audio, pyaudio.paContinue

        audio = self.rb.read(self._processing_size)
        chans = []
        chans.append(denoise(audio[:, 0]))
        chans.append(denoise(audio[:, 1]))

        return numpy.column_stack(chans).astype(self.dtype).tobytes(), pyaudio.paContinue

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
    SS = StreamSampler(buffer_delay=0)
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
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
    close()  # clean up the program runtime when the user closes the window
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
    close()  # clean up the program runtime when the user closes the window
