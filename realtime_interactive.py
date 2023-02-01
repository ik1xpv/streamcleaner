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
#realtime_interactive.py 1.0 - 1/19/23 - it works.

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
#pip install pipwin dearpygui pyroomacoustics opencv-python matplotlib
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

import pyroomacoustics as pra
from threading import Thread

import pyaudio
import dearpygui.dearpygui as dpg

import matplotlib.image as mimage
import matplotlib.colors as colors
from matplotlib import cm
import cv2

NBINS_GUI = int(48) # GUI displays nbins
N_BINS = int(36)
 
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
    fprint = numpy.linspace(0.0,1.0,points)
    fprint[1:-1]  /= 1 - fprint[1:-1]
    fprint[1:-1]  = numpy.log(fprint[1:-1])
    fprint[-1] = ((2*fprint[-2])  - fprint[-3]) 
    fprint[0] = -fprint[-1]
    return numpy.interp(fprint, (fprint[0], fprint[-1]),  (0, 1))

def generate_logit_window(size,sym =True):
  if sym == False or size<32:
    print("not supported")
    return numpy.zeros(size)
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

def logit_generation(npoints):
        logit = numpy.linspace(0, 1, npoints, dtype=numpy.float64)
        logit[1:-1] /= 1 - logit[1:-1]       # all but first and last, divide by 1 - the value.
        logit[1:-1] = numpy.log(logit[1:-1])  # all but the first and last, log(value)
        logit[-1] = ((2 * logit[-2]) - logit[-3])
        logit[0] = -logit[-1]
        m = logit[-1]
        logit += m
        logit /= m * 2
        return logit
#note: if you alter the number of bins, you need to regenerate this array. currently set to consider N_BINS bins        
LOGIT = logit_generation(N_BINS)        

@numba.njit(numba.float64[:](numba.float64[:,:]))
def fast_entropy(data: numpy.ndarray):
   entropy = numpy.zeros(data.shape[1],dtype=numpy.float64)
   for each in numba.prange(data.shape[1]):
      d = data[:,each]
      d = numpy.interp(d, (d[0], d[-1]), (0, +1))
      entropy[each] = 1 - numpy.corrcoef(d, LOGIT)[0,1]
   return entropy


@numba.jit(numba.float64[:,:](numba.float64[:,:],numba.int32[:],numba.float64,numba.float64[:]))
def fast_peaks(stft_:numpy.ndarray,entropy:numpy.ndarray,thresh:numpy.float32,entropy_unmasked:numpy.ndarray):
    #0.01811 practical lowest - but the absolute lowest is 0. 0 is a perfect logistic. 
    #0.595844362 practical highest - an array of zeros with 1 value of 1.
    #note: this calculation is dependent on the size of the logistic array.
    #if you change the number of bins, this constraint will no longer hold and renormalizing will fail.
    #a lookup table is not provided, however the value can be derived by setting up a one-shot fast_entropy that only considers one array,
    #sorting, interpolating, and comparing it. 
    mask = numpy.zeros_like(stft_) 
    for each in numba.prange(stft_.shape[1]):
        data = stft_[:,each].copy()
        if entropy[each] == 0:
            mask[0:N_BINS,each] =  0
            continue #skip the calculations for this row, it's masked already
        constant = atd(data) + man(data)  #by inlining the calls higher in the function, it only ever sees arrays of one size and shape, which optimizes the code
        test = entropy_unmasked[each]  / 0.6091672572096941 #currently set for N_BINS bins
        test = abs(test - 1) 
        thresh1 = (thresh*test)
        if numpy.isnan(thresh1):
            thresh1 = constant #catch errors
        constant = (thresh1+constant)/2
        data[data<constant] = 0
        data[data>0] = 1
        mask[0:N_BINS,each] = data[:]
    return mask

@numba.njit(numba.int32(numba.int32[:]))
def longestConsecutive(nums: numpy.ndarray):
        longest_streak = 0
        streak = 0
        prevstreak = 0
        for num in range(nums.size):
          if nums[num] == 1:
            streak += 1
          if nums[num] == 0:
            prevstreak = max(streak,prevstreak)
            streak = 0
        return max(streak,prevstreak)

import copy

@numba.njit(numba.int32[:](numba.int32[:]))
def smoothgaps(nums: numpy.ndarray):
      returns = nums.copy()
      if returns[1] == 1 and returns[0] == 0:
           returns[0] = 1
      if returns[-1] == 0 and returns[-2] ==1:
        returns[-1] = 1
      if returns[1] == 0 and returns[0] == 1:
           returns[0] = 0
      if returns[-1] == 1 and returns[-2] ==0:
        returns[-1] = 0
      for num in range(1,nums.size-1):
         if nums[num] == 0 and nums[num-1] == 1 and nums[num+1] ==1:
           returns[num] = 1
      for num in range(1,nums.size-1):
         if returns[num] == 1 and returns[num-1] == 0 and returns[num+1] ==0:
           returns[num] = 0


      return returns

def update_gui( stft_in, stft_out):
    #generates the GUI images 
    scale = 0.1
    stft_in= stft_in*scale   
    stft_out= stft_out*scale
    
    arr_color_d = cm.ScalarMappable(cmap="turbo").to_rgba(stft_in[:-(257-NBINS_GUI)], bytes=False, norm=False) #only the first 4k span
    arr_color_d = numpy.flipud(arr_color_d) #updown freq axis
    arr_color_d = cv2.resize(arr_color_d, dsize=(256, NBINS_GUI*2), interpolation=cv2.INTER_CUBIC)
    
    arr_color = cm.ScalarMappable(cmap="turbo").to_rgba(stft_out[:-(257-NBINS_GUI)], bytes=False, norm=False) #only the first 4k span
    arr_color = numpy.flipud(arr_color) #updown freq axis
    arr_color = cv2.resize(arr_color, dsize=(256, NBINS_GUI*2), interpolation=cv2.INTER_CUBIC)
    
    dpg.set_value("dirty_texture", arr_color_d) 
    dpg.set_value("clean_texture", arr_color)
    return
    
def mask_generation(stft_vh1:numpy.ndarray,stft_vl1: numpy.ndarray,NBINS:int):

    #24000/256 = 93.75 hz per frequency bin.
    #a 4000 hz window(the largest for SSB is roughly 43 bins.
    #https://en.wikipedia.org/wiki/Voice_frequency
    #however, practically speaking, voice frequency cuts off just above 3400hz.
    #*most* SSB channels are constrained far below this.
    #to catch most voice activity on shortwave, we use the first 32 bins, or 3000hz.
    #we automatically set all other bins to the residue value.
    #reconstruction or upsampling of this reduced bandwidth signal is a different problem we dont solve here.
    stft_vh = numpy.ndarray(shape=stft_vh1.shape, dtype=numpy.float64, order='C') 
    stft_vl = numpy.ndarray(shape=stft_vh1.shape, dtype=numpy.float64,order='C') 
    stft_vh[:] = copy.deepcopy(stft_vh1)
    stft_vl[:] = copy.deepcopy(stft_vl1)


    lettuce_euler_macaroni = 0.057
    stft_vs = numpy.sort(stft_vl[0:N_BINS,:],axis=0) #sort the array
    entropy_unmasked = fast_entropy(stft_vs)
    entropy_unmasked[numpy.isnan(entropy_unmasked)] = 0
    entropy = smoothpadded(entropy_unmasked,14).astype(dtype=numpy.float64)

    factor = numpy.max(entropy)

    if factor < lettuce_euler_macaroni: 
      return (stft_vh[:,64:128] * 1e-5).T


    entropy[entropy<lettuce_euler_macaroni] = 0
    entropy[entropy>0] = 1
    entropy = entropy.astype(dtype=numpy.int32)

    criteria_before = 1
    criteria_after = 1

    entropy_minimal = entropy[64-32:128+32] #concluded 
    nbins = numpy.sum(entropy_minimal)
    maxstreak = longestConsecutive(entropy_minimal)
    if nbins<22 and maxstreak<16:
      return (stft_vh[:,64:128]  * 1e-5).T
    #what this will do is simply reduce the amount of work slightly

    entropy = smoothgaps(entropy)
    #remove anomalies

    mask=numpy.zeros_like(stft_vh)

    stft_vh1 = stft_vh[0:N_BINS,:]
    thresh = threshold(stft_vh1[stft_vh1>man(stft_vl[0:N_BINS,:].flatten())])/2

    mask[0:N_BINS,:] = fast_peaks(stft_vh[0:N_BINS,:],entropy,thresh,entropy_unmasked)
    mask = mask[:,(64-16):(128+16)]
    #doing a bit of cropping delivers identical results with less cycles used.
    mask = numpy_convolve_filter_longways(mask,5,17)
    mask = mask[:,16:(64+16)] #further crop to the final mask size
    mask2 = numpy_convolve_filter_topways(mask,5,2) 
    mask2 = numpy.where(mask==0,0,mask2)

    #i think normalizing, is risky, because we're only taking the localized frame maximum.
    #this stands to risk causing some small fragments to be enhanced too much and perhaps distorting the results.
    #additionally, what's the maximum? it's *never* above 1.0, usually around 0.75-0.92
    #adding a gain stage *after* filtration is generally a better idea than trying to compensate for it here.
    #the choice of filler value 1e-5 is unknown. 
    mask2[mask2<1e-5] = 1e-5

    return mask2.T

class FilterThread(Thread):
    def __init__(self):
        super(FilterThread, self).__init__()
        self.running = True
        self.NFFT = 512
        self.NBINS=N_BINS
        self.hop = 128
        self.hann = generate_hann(self.NFFT)
        self.logistic = generate_logit_window(self.NFFT)
        self.synthesis = pra.transform.stft.compute_synthesis_window(self.hann, self.hop)
        self.stft = pra.transform.STFT(512, hop=self.hop, analysis_window=self.hann,synthesis_window=self.synthesis ,online=True)
        self.oneshot_hann = pra.transform.STFT(512, hop=self.hop, analysis_window=self.hann,synthesis_window=self.synthesis ,online=True)
        self.oneshot_logit = pra.transform.STFT(512, hop=self.hop, analysis_window=self.logistic,synthesis_window=self.synthesis ,online=True)
        self.audio = numpy.zeros(8192*3)

    def process(self,data,):        
           self.audio = numpy.roll(self.audio,-8192)
           self.audio[-8192:] = data[:]
           logit = self.oneshot_logit.analysis(self.audio)
           hann = self.oneshot_hann.analysis(self.audio)
           self.stft.analysis(self.audio[8192:-8192])
           stft_in = numpy.abs(self.stft.X.T)
           mask = mask_generation(numpy.abs(hann).T,numpy.abs(logit).T,self.NBINS)
           #output = self.stft.synthesis(self.stft.X* mask)
           stft_mask = (self.stft.X* mask)
           stft_out = numpy.abs(stft_mask.T)       
           output = self.stft.synthesis(stft_mask)
           update_gui(stft_in, stft_out)
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
        self.denoise = True
    

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
        if self.denoise == True:
            chans.append(self.rightthread.process(audio_in[:,0]))
            chans.append(self.leftthread.process(audio_in[:,1]))
        else:
            chans.append(audio_in[:,0])
            chans.append(audio_in[:,1])
            dpg.set_value("dirty_texture", dirtytexture) 
            dpg.set_value("clean_texture", dirtytexture) 
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

def denoisetoggle(sender, app_data,user_data):
        if SS.denoise == True:
            dpg.set_item_label("toggleswitch", "cleanup is OFF")
            SS.denoise = False
        else:
            dpg.set_item_label("toggleswitch", "cleanup is ON ")
            SS.denoise = True



if __name__ == "__main__":
    SS = StreamSampler()
    SS.listen()
    def close():
        dpg.destroy_context()
        SS.stop()
        quit()


    dpg.create_context()   

    cleantexture = [0.1, 0.1, 0, 1] * 256 * NBINS_GUI *2
    dirtytexture = [0.2, 0, 0.2, 1] * 256 * NBINS_GUI *2
    #patch from joviex- the enumeration in the online docs showing .append doesn't work for larger textures        
    with dpg.texture_registry():
        dpg.add_dynamic_texture(256, NBINS_GUI *2, dirtytexture, tag="dirty_texture")
    with dpg.texture_registry():
        dpg.add_dynamic_texture(256, NBINS_GUI *2, cleantexture, tag="clean_texture")
        
    dpg.create_viewport(title= 'realtime_interactive 1.0',width=290, height=300)
    with dpg.window(label= 'stft', width=278, height=260):
        dpg.add_image("dirty_texture")
        dpg.add_image("clean_texture")
        dpg.add_button(label="cleanup is ON", tag="toggleswitch", callback=denoisetoggle)
    dpg.setup_dearpygui()
    dpg.configure_app(auto_device=True)
    
    dpg.show_viewport()
    dpg.start_dearpygui()
    while dpg.is_dearpygui_running(): 
        dpg.render_dearpygui_frame()   
    close()  # clean up the program runtime when the user closes the window
