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

#idea and code and bugs by Joshuah Rainstar   : 
#fork, mod by Oscar Steila : https://groups.io/g/NextGenSDRs/topic/spectral_denoising
#cleanup_classic1.1.1.py


#12/5/2022 : For best results, i recommend combining this with dynamic expansion and compression.

#This experiment is intended to explore improvements to common threshholding and denoising methods in a domain
#which is readily accessible for experimentation, namely simple scripting without any complex languages or compilers.
#Python's performance has come a long way and with numpy, numba, and a little optimization, it is closer to c++.
#Consider this a challenge. Can you do better or improve this algorithm?. 

#Feel free to try your own algorithms and optimizations.

#In particular, thresholding is set to provide a safe denoising basis which will catch *most* speech.
#entropy and constants are used to provide a statistically robust(but perhaps inaccurate) noise measure.
#these are calculated over a window of 0~2900hz, which considers the strongest voice components. 

#My efforts were originally inspired by the noisereduce python library, which works somewhat well but was designed for bird calls.
#noisereduce assumes that the signal is above the noise floor, attempts to estimate the noise floor, and then
#calculates a threshold high enough to identify the voice formant waveform ridges, which is then softened to form
#a "contour" around the signal. 

#the cleanup algorithm assumes first that there is a fundemental floor below the noise and that more robust thresholding can be used
#to find signal peaks. It also uses a different approach to softening the mask.  Cleanup also works well on CW and some data modes, 
#but is only intended for listening-beware of results data wise.
#Science has concluded the harsher the noise, the more it stresses the listener. It can be harmful to health.
#Therefore the use of a denoising plugin which is effective is beneficial, and improves intelligability. 
#Cleanup does not yet attempt to recover signal.


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
#pip install librosa pipwin dearpygui np-rw-buffer
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
#Please put other denoising methods *after* this method.

import os
import numpy
import numpy as np

import pyaudio
from librosa import stft,istft

from time import sleep
from np_rw_buffer import AudioFramingBuffer
import dearpygui.dearpygui as dpg


def padarray(A, size):
    t = size - len(A)
    return numpy.pad(A, pad_width=(0, t), mode='constant',constant_values=numpy.std(A))

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def fast_hamming(M, sym=True):
    a = [0.5, 0.5]
    fac = numpy.linspace(-numpy.pi, numpy.pi, M)
    w = numpy.zeros(M)
    for k in range(len(a)):
        w += a[k] * numpy.cos(k * fac)
    return w

def boxcar(M, sym=True):
    a = [0.5, 0.5]
    fac = numpy.linspace(-numpy.pi, numpy.pi, M)
    w = numpy.zeros(M)
    for k in range(len(a)):
        w += a[k] * numpy.cos(k * fac)
        return w

def hann(M, sym=True):
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


def corrected_logit(size):
    fprint = numpy.linspace(0, 1, size)
    fprint [1:-1] /= 1 - fprint [1:-1]
    fprint [1:-1] = numpy.log(fprint [1:-1])
    fprint[0] = -6
    fprint[-1] = 6
    return fprint

#precalculate the logistic function for our entropy calculations.
#save some cycles with redundancy.
#since we only calculate entropy over a fixed window, we can store our logit.
logit = corrected_logit(32)

def entropy(data: numpy.ndarray):
    a = numpy.sort(data)
    scaled = numpy.interp(a, (a[0], a[-1]), (-6, +6))
    z = numpy.corrcoef(scaled, logit)
    completeness = z[0, 1]
    sigma = 1 - completeness
    return sigma



def runningMeanFast(x, N):
    return numpy.convolve(x, numpy.ones((N,))/N,mode="valid")

#depending on presence of openblas, as fast as numba.  
def numpy_convolve_filter(data: numpy.ndarray):
   normal = data.copy()
   transposed = data.copy()
   transposed = transposed.T
   transposed_raveled = numpy.ravel(transposed)
   normal_raveled = numpy.ravel(normal)

   A =  runningMeanFast(transposed_raveled, 3)
   transposed_raveled[0] = (transposed_raveled[0] + (transposed_raveled[1] + transposed_raveled[2]) / 2) /3
   transposed_raveled[-1] = (transposed_raveled[-1] + (transposed_raveled[-2] + transposed_raveled[-3]) / 2)/3
   transposed_raveled[1:-1] = A 
   transposed = transposed.T


   A =  runningMeanFast(normal_raveled, 3)
   normal_raveled[0] = (normal_raveled[0] + (normal_raveled[1] + normal_raveled[2]) / 2) /3
   normal_raveled[-1] = (normal_raveled[-1] + (normal_raveled[-2] + normal_raveled[-3]) / 2)/3
   normal_raveled[1:-1] = A
   return (transposed + normal )/2


def numpyfilter_wrapper_50(data: numpy.ndarray):
  d = data.copy()
  for i in range(50):
    d = numpy_convolve_filter(d)
  return d


def denoise_old(data: numpy.ndarray):
    data= numpy.asarray(data,dtype=float) #correct byte order of array   

    stft_r = stft(data,n_fft=512,window=boxcar) #get complex representation
    stft_vr = numpy.square(stft_r.real) + numpy.square(stft_r.imag) #obtain absolute**2
    Y = stft_vr[~numpy.isnan(stft_vr)]
    max = numpy.where(numpy.isinf(Y),0,Y).argmax()
    max = Y[max]
    stft_vr = numpy.nan_to_num(stft_vr, copy=True, nan=0.0, posinf=max, neginf=0.0)#correct irrationalities
    stft_vr = numpy.sqrt(stft_vr) #stft_vr >= 0 
    stft_vr=(stft_vr-numpy.nanmin(stft_vr))/numpy.ptp(stft_vr)
    ent1 = numpy.apply_along_axis(func1d=entropy,axis=0,arr=stft_vr[0:32,:]) #32 is pretty much the speech cutoff?
    ent1 = ent1 - numpy.min(ent1)

    t = threshhold(stft_vr)     
    mask_one = numpy.where(stft_vr>=t, 1,0)
    stft_demo = numpy.where(mask_one == 0, stft_vr,0)
    stft_d = stft_demo.flatten()
    stft_d = stft_d[stft_d>0]
    r = man(stft_d) #obtain a noise background basis
    
    stft_r = stft(data,n_fft=512,window=hann) #get complex representation
    
    stft_vr = numpy.square(stft_r.real) + numpy.square(stft_r.imag) #obtain absolute**2
    Y = stft_vr[~numpy.isnan(stft_vr)]
    max = numpy.where(numpy.isinf(Y),-numpy.Inf,Y).argmax()
    max = Y[max]
    stft_vr = numpy.nan_to_num(stft_vr, copy=True, nan=0.0, posinf=max, neginf=0.0)#correct irrationalities
    stft_vr = numpy.sqrt(stft_vr) #stft_vr >= 0 
    stft_vr=(stft_vr-numpy.nanmin(stft_vr))/numpy.ptp(stft_vr) #normalize to 0,1
   
    ent = numpy.apply_along_axis(func1d=entropy,axis=0,arr=stft_vr[0:32,:]) #32 is pretty much the speech cutoff?
    ent = ent - numpy.min(ent)
    ent  = moving_average(ent,14)
    ent1  = moving_average(ent1,14)
    #seems to be a reasonable compromise
    minent = numpy.minimum(ent,ent1)
    minent=(minent-numpy.nanmin(minent))/numpy.ptp(minent)#correct basis
    maxent = numpy.maximum(ent,ent1)
    
    trend = moving_average(maxent,20)
    factor = numpy.max(trend)
    if factor < 0.0577215664: 

      stft_r = stft_r * r
      processed = istft(stft_r,window=hann)
      return processed
      #no point wasting cycles smoothing information which isn't there!

    maxent=(maxent-numpy.nanmin(maxent))/numpy.ptp(maxent)#correct basis 

    ent = (maxent+minent)
    ent = ent - numpy.min(ent)
    trend=(ent-numpy.nanmin(ent))/numpy.ptp(ent)#correct basis 

    t1 = atd(trend)/2 #unclear where to set this. too aggressive and it misses parts of syllables.
    trend[trend<t1] = 0
    trend[trend>0] = 1
    t = (threshhold(stft_vr[stft_vr>=t]) - atd(stft_vr[stft_vr>=t]) ) +man(stft_vr)   #obtain the halfway threshold
    #note: this threshhold is still not perfectly refined, but had to be optimized for a variety of SNR.
    mask_two = numpy.where(stft_vr>=t, 1.0,0)

    mask = mask_two * trend[None,:] #remove regions from the mask that are noise
    r = r * factor
    mask[mask==0] = r #reduce warbling, you could also try r/2 or r/10 or something like that, its not as important
    mask = numpyfilter_wrapper_50(mask)
    
    mask=(mask-numpy.nanmin(mask))/numpy.ptp(mask)#correct basis    

    stft_r = stft_r * mask
    processed = istft(stft_r,window=hann)
    return processed

import numba
@numba.jit()
def entropy_numba(data: numpy.ndarray):
    a = numpy.sort(data)
    scaled = numpy.interp(a, (a[0], a[-1]), (-6, +6))
    z = numpy.corrcoef(scaled, logit)
    completeness = z[0, 1]
    sigma = 1 - completeness
    return sigma


def numpyentropycheck(data: numpy.ndarray):
  d = data.copy()
  raveled = numpy.ravel(d)
  windows =  numpy.lib.stride_tricks.sliding_window_view(raveled, window_shape = 32)
  raveled[0:windows.shape[0]] = numpy.apply_along_axis(func1d = entropy_numba,axis=1,arr=windows)
  return d  

#here is an experimental new denoising algorithm. 
#it may be less sensitive than the previous version, but it is computationally competitive,
#and does more. It may be more robust. It may be more statistically valid.

def denoise(data: numpy.ndarray):
    #0.0747597920253411435178730 #maximum sensitivity  at this constant. this is the parking constant.
    #0.0834626841674073186814297  maximum denoise at this constant. This is the AGM.
    # set the constant somewhere between the two to fine-tune the noise sensitivity.
    sensitivity_constant = (0.0834626841674073186814297 + 0.0747597920253411435178730)/2

    data= numpy.asarray(data,dtype=float) #correct byte order of array   

    stft_r = stft(data,n_fft=512,window=boxcar) #get complex representation
    stft_vr =  numpy.abs(stft_r) #returns the same as other methods
    stft_vr=(stft_vr-numpy.nanmin(stft_vr))/numpy.ptp(stft_vr) #normalize to 0,1
    window = stft_vr[0:32,:]
    window = numpy.pad(window,((0,0),(32,32)),mode="symmetric")
    e = numpyentropycheck(window.T).T[:,32:-32]
    entropy = numpy.apply_along_axis(func1d=numpy.max,axis=0,arr=e)
    o = numpy.pad(entropy, entropy.size//2, mode='median')
    entropy = moving_average(o,14)[entropy.size//2: -entropy.size//2]
    factor = numpy.sum(entropy)/entropy.size
    floor = threshhold(stft_vr)  #use the floor from the boxcar

    stft_r = stft(data,n_fft=512,window=hann) #get complex representation
    stft_vr =  numpy.abs(stft_r) #returns the same as other methods

    stft_vr=(stft_vr-numpy.nanmin(stft_vr))/numpy.ptp(stft_vr) #normalize to 0,1
    residue = man(stft_vr)  
    if factor < 0.0747597920253411435178730:  #Renyi's parking constant m 
      stft_r = stft_r * residue #return early, and terminate the noise
      processed = istft(stft_r,window=hann)
      return processed 


    entropy_threshhold = sensitivity_constant 

    entropy[entropy<entropy_threshhold] = 0
    entropy[entropy>0] = 1

    threshold = (threshhold(stft_vr[stft_vr>=floor]) - atd(stft_vr[stft_vr>=floor]))
    mask_two = numpy.where(stft_vr>=threshold, 1.0,0)


    mask = mask_two * entropy[None,:] #remove regions from the mask that are noise
    residue = residue * factor
    mask[mask==0] = residue 
    mask = numpyfilter_wrapper_50(mask)
    
    mask=(mask-numpy.nanmin(mask))/numpy.ptp(mask)#correct basis    

    stft_r = stft_r * mask
    processed = istft(stft_r,window=hann)
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
