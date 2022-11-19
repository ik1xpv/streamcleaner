#idea and code by Joshuah Rainstar   : https://groups.io/g/NextGenSDRs/message/1085
#fork, mods and bugs by Oscar Steila : https://groups.io/g/NextGenSDRs/topic/spectral_denoising
#cleanup_classic1.0.3.2.py
#19/11/2022#Warning it's an experiment early version that shows input and output stft spectrums on screen.    
#
#11/15/2022#How to use this file:
#you will need 1 virtual audio cable- try https://vb-audio.com/Cable/ if you use windows.
#install and configure the virtual audio cable and your speakers for 16 bits, 48000hz, two channels.
#if you use OSX or Linux, you will have to modify this file by changing audio devices, settings and libraries appropriately.
#step one: put this file somewhere and remember where it is.
#step two: using https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Windows-x86_64.exe
#install python.
#step three: locate the dedicated python terminal in your start menu, called mambaforge prompt.
#within that prompt, give the following instructions:
#conda install pip numpy
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
#and 10db of gain after this script, followed by a compressor- if you know how to use one.

import os
import numpy
import numba
import pyaudio
import librosa

from time import sleep
from np_rw_buffer import AudioFramingBuffer
import dearpygui.dearpygui as dpg
import array

import matplotlib.image as mimage
import matplotlib.colors as colors
from matplotlib import cm
import cv2


def padarray(A, size):
    t = size - len(A)
    return numpy.pad(A, pad_width=(0, t), mode='constant')

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

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
    #sigma = 5.0
    #var = sigma * sigma
    k = 12 #int(3.0 * 5 + 0.5)
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
    horiz = horiz.transpose((1, 0)).astype(x.dtype)
    return horiz

@numba.jit(numba.float64[:,:](numba.float64[:,:]),cache=True)
def mask_t(stft_vr: numpy.ndarray):
    stft_vr = numpy.sqrt(stft_vr)
    stft_vr = numpy.where(stft_vr>0,stft_vr,0)
    for each in range(stft_vr.shape[0]):
        for a in range(stft_vr.shape[1]):
          if stft_vr[each,a] > 0:
            stft_vr[each,a] = numpy.log10(stft_vr[each,a])
    stft_vr /= numpy.max(stft_vr)
    stft_avg = numpy.zeros(stft_vr.shape[1],dtype=numpy.float64)

    for each in range(stft_vr.shape[1]):
      samples = stft_vr[:,each]
      stft_avg[each] = numpy.max(samples)
      samples /= numpy.max(samples)
      samples = numpy.where(samples>0,samples,0)
      samples =  numpy.power(10,samples)
      samples /= numpy.max(samples)
      samples = numpy.where(samples>0,samples,0)
      stft_vr[:,each] = samples
    trend = stft_avg
    trend = numpy.where(trend>0,trend,0)#get rid of the inter-frame glitch 
    trend =  numpy.power(10,trend)
    trend /= numpy.max(trend)
    trend = numpy.where(trend>0,trend,0)
    for each in range(stft_vr.shape[1]):
      stft_vr[:,each] = stft_vr[:,each] * trend[each]
    return stft_vr

#do not use this as a window for a reversable STFT! it will NOT WORK
def broken_hamming(M, sym=True):
    a = [0.5, 0.5]
    fac = numpy.linspace(-numpy.pi, numpy.pi, M)
    w = numpy.zeros(M)
    for k in range(len(a)):
        w += a[k] * numpy.cos(k * fac)
        return w

def fast_hamming(M, sym=True):
    a = [0.5, 0.5]
    fac = numpy.linspace(-numpy.pi, numpy.pi, M)
    w = numpy.zeros(M)
    for k in range(len(a)):
        w += a[k] * numpy.cos(k * fac)
        return w #never use this except for lower threshold

PCS = numpy.ones(257)      # Perceptual Contrast Stretching
PCS[0:3] = 1
PCS[3:6] = 1.070175439
PCS[6:9] = 1.182456140
PCS[9:12] = 1.287719298
PCS[12:138] = 1.4       # Pre Set
PCS[138:166] = 1.322807018
PCS[166:200] = 1.238596491
PCS[200:241] = 1.161403509
PCS[241:256] = 1.077192982

DENOISE = True
#STOP4KHZ = int((512 *8000)/48000)
NBIN = int((512 *8000)/48000) # 8000 Hz spectrogram span
L_OFS = 1.2 # log offset

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    med = numpy.median(arr)
    return numpy.median(numpy.abs(arr - med))

def atd(arr):
    med = numpy.median(arr)
    return numpy.sqrt(mad(arr))

def aSNR(arr):
  return mad(arr)/atd(arr)

def SNR(arr):
  return numpy.mean(arr)/numpy.std(arr)

def threshhold(arr):
  return (atd(arr)+ numpy.median(arr)) * (SNR(arr) - aSNR(arr))

def denoise(data: numpy.ndarray,DENOISE):
    data= numpy.asarray(data,dtype=float) #correct byte order of array
    stft_bad = librosa.stft(data,n_fft=512,window=broken_hamming) #get complex representation
    stft_vr = numpy.square(abs(stft_bad.real)) + numpy.square(abs(stft_bad.imag)) #obtain absolute
    stft_vr = numpy.sqrt(stft_vr) #shouldnt return anything weird
    Y = stft_vr[~numpy.isnan(stft_vr)]
    max = numpy.where(numpy.isinf(Y),-numpy.Inf,Y).argmax()
    min = numpy.where(numpy.isinf(Y),-numpy.Inf,Y).argmin()
    max = Y[max]
    min = Y[min]
    stft_vr = numpy.nan_to_num(stft_vr, copy=True, nan=0.0, posinf=max, neginf=min)#correct irrationalities
    stft_vr = (stft_vr - numpy.min(stft_vr))/numpy.ptp(stft_vr) #normalize to 0,1
    t = threshhold(stft_vr[stft_vr>0])  #obtain the noise threshold

    stft_r = librosa.stft(data,n_fft=512,window=fast_hamming) #get complex representation
    stft_vr = numpy.square(abs(stft_r.real)) + numpy.square(abs(stft_r.imag)) #obtain absolute
    stft_vr = numpy.sqrt(stft_vr) #shouldnt return anything weird
    Y = stft_vr[~numpy.isnan(stft_vr)]
    max = numpy.where(numpy.isinf(Y),-numpy.Inf,Y).argmax()
    min = numpy.where(numpy.isinf(Y),-numpy.Inf,Y).argmin()
    max = Y[max]
    min = Y[min]
    stft_vr = numpy.nan_to_num(stft_vr, copy=True, nan=0.0, posinf=max, neginf=min)#correct irrationalities
    stft_vr = (stft_vr - numpy.min(stft_vr))/numpy.ptp(stft_vr) #normalize to 0,1

    mask_one = numpy.zeros(stft_vr.shape,dtype=int)
    mask_one = numpy.where(stft_vr>t, 1,0)

    mask_two = numpy.zeros(stft_vr.shape,dtype=int)

    stft_demo= numpy.where(mask_one == 1, stft_vr,0)
    t = threshhold(stft_demo[stft_demo>0])  #obtain the halfway threshold
  
    mask_two = numpy.zeros(stft_vr.shape,dtype=float)

    mask_two = numpy.where(stft_vr>=t, 1,0)
    mask_two_ = filter(mask_two)
    mask_two_[mask_two == 1] = 1
    if DENOISE: 
         stft_r = stft_r * mask_two_#apply mask
         processed = librosa.istft(stft_r,window=fast_hamming)
    else:
         processed = data

    arr_color = cm.ScalarMappable(cmap="turbo").to_rgba(stft_vr, bytes=False,norm=True)
    arr_color = cv2.resize(arr_color, dsize=(660, 257), interpolation=cv2.INTER_CUBIC)
    dpg.set_value("dirty_texture", arr_color) 

   
    stft_vr = numpy.square(abs(stft_r.real)) + numpy.square(abs(stft_r.imag)) #obtain absolute
    stft_vr = numpy.sqrt(stft_vr) #shouldnt return anything weird
    Y = stft_vr[~numpy.isnan(stft_vr)]
    max = numpy.where(numpy.isinf(Y),-numpy.Inf,Y).argmax()
    min = numpy.where(numpy.isinf(Y),-numpy.Inf,Y).argmin()
    max = Y[max]
    min = Y[min]
    stft_vr = numpy.nan_to_num(stft_vr, copy=True, nan=0.0, posinf=max, neginf=min)#correct irrationalities
    stft_vr = (stft_vr - numpy.min(stft_vr))/numpy.ptp(stft_vr) #normalize to 0,1


    arr_color = cm.ScalarMappable(cmap="turbo").to_rgba(stft_vr, bytes=False,norm=True)
    arr_color = cv2.resize(arr_color, dsize=(660, 257), interpolation=cv2.INTER_CUBIC)
    dpg.set_value("clean_texture", arr_color)
    return processed


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
        self.denoise = True

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
        chans.append(denoise(audio[:, 0],self.denoise))
        chans.append(denoise(audio[:, 1],self.denoise))
        dpg.set_value("plot2", chans[0])
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

    def denoisetoggle(sender, app_data,user_data):
        if SS.denoise.enabled == True:
            dpg.set_item_label("toggleswitch", "denoiser is ON")
            SS.denoise = False
        else:
            dpg.set_item_label("toggleswitch", "denoiser is OFF")
            SS.denoise = True


    cleantexture = [1, 1, 0, 1] * 660 * 257
    dirtytexture = [1, 0, 1, 1] * 660 * 257
    #patch from joviex- the enumeration in the online docs showing .append doesn't work for larger textures        
    with dpg.texture_registry():
        dpg.add_dynamic_texture(660, 257, dirtytexture, tag="dirty_texture")
    with dpg.texture_registry():
        dpg.add_dynamic_texture(660, 257, cleantexture, tag="clean_texture")
    dpg.create_viewport(title= 'Denoiser',width=660, height=780)
    dpg.setup_dearpygui()


    with dpg.window(label= 'cleanup_classic.1.0.4.0', width=660, height=780):
        dpg.add_text("stft input")
        dpg.add_image("dirty_texture")
        dpg.add_text("stft output")
        dpg.add_image("clean_texture")
        dpg.add_text("waveform out")
        dpg.add_simple_plot( min_scale=-1.0, max_scale=1.0, width=660, height=100, tag="plot2") 
        dpg.add_button(label="Disable", tag="toggleswitch", callback=denoisetoggle)
        


    dpg.configure_app(auto_device=True)

    dpg.show_viewport()
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

        
    close()  # clean up the program runtime when the user closes the window
   