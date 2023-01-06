print("this is only an experiment and does not reflect the performance of other scripts located in this directory")
#also this instantly crashes when called with 4096 for some obscene reason
exit(0)
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
#realtime.py 1.6.23


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
#pip install pipwin dearpygui pyroomacoustics
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
os.environ['SSQ_PARALLEL'] = '0'



def generate_hann(M, sym=True):
    a = [0.5, 0.5]
    fac = numpy.linspace(-numpy.pi, numpy.pi, M)
    w = numpy.zeros(M)
    for k in range(len(a)):
        w += a[k] * numpy.cos(k * fac)
    return w

@numba.jit()
def man(arr):
    med = numpy.nanmedian(arr[numpy.nonzero(arr)])
    return numpy.nanmedian(numpy.abs(arr - med))

@numba.jit()
def atd(arr):
    x = numpy.square(numpy.abs(arr - man(arr)))
    return numpy.sqrt(numpy.nanmean(x))

#@numba.jit()
#def threshhold(arr):
#  return (atd(arr)+ numpy.nanmedian(arr[numpy.nonzero(arr)])) 

def runningMeanFast(x, N):
    return numpy.convolve(x, numpy.ones((N,))/N,mode="valid")

def moving_average(x, w):
    return numpy.convolve(x, numpy.ones(w), 'same') / w

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

def moving_average(x, w):
    return numpy.convolve(x, numpy.ones(w), 'same') / w

def smoothpadded(data: numpy.ndarray):
  o = numpy.pad(data, data.size//2, mode='median')
  return moving_average(o,14)[data.size//2: -data.size//2]

@numba.jit()
def threshold(data: numpy.ndarray):
 return numpy.sqrt(numpy.nanmean(numpy.square(numpy.abs(data -numpy.nanmedian(numpy.abs(data - numpy.nanmedian(data[numpy.nonzero(data)]))))))) + numpy.nanmedian(data[numpy.nonzero(data)])

def generate_true_logistic(points):
    fprint = numpy.linspace(0.0,1.0,points)
    fprint[1:-1]  /= 1 - fprint[1:-1]
    fprint[1:-1]  = numpy.log(fprint[1:-1])
    fprint[-1] = ((2*fprint[-2])  - fprint[-3]) 
    fprint[0] = -fprint[-1]
    return numpy.interp(fprint, (fprint[0], fprint[-1]),  (0, 1))

def generate_logit(size,sym =True):
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

@numba.njit()
def fast_entropy(data: numpy.ndarray,logit):
   entropy = numpy.zeros(data.shape[1])
   for each in range(data.shape[0]):
      d = data[each,:]
      d = numpy.interp(d, (d[0], d[-1]), (0, +1))
      entropy[each] = 1 - numpy.corrcoef(d, logit)[0,1]
   return entropy


@numba.jit()
def fast_peaks(stft_:numpy.ndarray,entropy:numpy.ndarray,thresh:numpy.float64,entropy_unmasked:numpy.ndarray,NBINS):
    #0.01811 practical lowest
    #0.595844362 practical highest
    mask = numpy.zeros_like(stft_)
    for each in range(stft_.shape[0]):
        data = stft_[each,:]
        if entropy[each] == 0:
            continue #skip the calculations for this row, it's masked already
        constant = atd(data) + man(data)  #by inlining the calls higher in the function, it only ever sees arrays of one size and shape, which optimizes the code
        test = entropy_unmasked[each]  / 0.595844362
        test = abs(test - 1) 
        thresh1 = (thresh*test)
        if numpy.isnan(thresh1):
            thresh1 = constant #catch errors
        constant = (thresh1+constant)/2
        data[data<constant] = 0
        data[data>0] = 1
        mask[each,0:NBINS] = data[0:NBINS]
    return mask


@numba.jit()
def threshhold(arr):
  return (atd(arr)+ numpy.nanmedian(arr[numpy.nonzero(arr)])) 

@numba.jit()
def longestConsecutive(nums):
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

def mask_generation(stft_vh,entropy_unmasked,NBINS):

    #24000/256 = 93.75 hz per frequency bin.
    #a 4000 hz window(the largest for SSB is roughly 43 bins.
    #https://en.wikipedia.org/wiki/Voice_frequency
    #however, practically speaking, voice frequency cuts off just above 3400hz.
    #*most* SSB channels are constrained far below this.
    #to catch most voice activity on shortwave, we use the first 32 bins, or 3000hz.
    #we automatically set all other bins to the residue value.
    #reconstruction or upsampling of this reduced bandwidth signal is a different problem we dont solve here.

    lettuce_euler_macaroni = 0.0596347362323194074341078499369279376074

    entropy = smoothpadded(entropy_unmasked)
    factor = numpy.max(entropy)

    if factor < lettuce_euler_macaroni: 
      return stft_vh[64:128,:]* 0

    entropy[entropy<lettuce_euler_macaroni] = 0
    entropy[entropy>0] = 1

    criteria_before = 1
    criteria_after = 1

    entropy_before = entropy[0:128]
    nbins = numpy.sum(entropy_before)
    maxstreak = longestConsecutive(entropy_before)
    if nbins<44 and maxstreak<32:
        criteria_before = 0
    entropy_after = entropy[64:]
    nbins = numpy.sum(entropy_before)
    maxstreak = longestConsecutive(entropy_before)
    if nbins<44 and maxstreak<32:
        criteria_after = 0


    #ok, so the shortest vowels are still over 100ms long. That's around 37.59 samples. Call it 38.
    #half of 38 is, go figure, 17.
    #now we do have an issue- what if a vowel is being pronounced in the middle of transition?
    #this *could* result in the end or beginning of words being truncated, but with this criteria,
    #we reasonably establish that there are no regions as long as half a vowel.
    #if there's really messed up speech(hence small segments) but enough of it(hence 22 bins)
    #then we can consider the frame to consist of speech
    
    # an ionosound sweep is also around or better than 24 samples, also
    if criteria_before ==0 and criteria_after == 0:
      return stft_vh[64:128,:]* 0
          
    mask=numpy.zeros_like(stft_vh)
    stft_vi = stft_vh[:,0:NBINS].flatten()
    thresh = threshold(stft_vi[stft_vi>man(stft_vi)]) #unknown if this is best
    mask[:] = fast_peaks(stft_vh,entropy,thresh,entropy_unmasked,NBINS)
     
    
    mask = numpyfilter_wrapper_50(mask)
    return mask[64:128,:]
    


class FilterThread(Thread):
    def __init__(self,  clearflag):
        super(FilterThread, self).__init__()
        self.running = True
        self.nframes = 64
        self.NFFT = 512
        self.NBINS=32
        self.hop = 64
        self.initialized = 0     
        self.clearflag = 0
        self.hann = generate_hann(self.NFFT)
        self.logistic = generate_logit(self.NFFT)
        self.synthesis = pra.transform.stft.compute_synthesis_window(self.hann, self.hop)
        self.stft = pra.transform.STFT(512, hop=self.hop, analysis_window=self.hann,synthesis_window=self.synthesis ,online=True,num_frames=self.nframes)
        self.oneshot_hann = pra.transform.STFT(512, hop=self.hop, analysis_window=self.hann,synthesis_window=self.synthesis ,online=False,num_frames=self.nframes)
        self.oneshot_logit = pra.transform.STFT(512, hop=self.hop, analysis_window=self.logistic,synthesis_window=self.synthesis ,online=False,num_frames=self.nframes)
        self.true_logistic  = generate_true_logistic(self.NBINS)

    def process(self,data,clear_flag:float = 0):        
        if clear_flag == 1 or self.initialized ==0:
           self.future = numpy.abs(self.oneshot_hann.analysis(data))
           logit = numpy.abs(self.oneshot_logit.analysis(data))
           self.entropy_future = fast_entropy(numpy.sort(logit[:,0:self.NBINS],axis=1),self.true_logistic)
           self.present = numpy.zeros_like(self.future)
           self.present_entropy = numpy.zeros_like(self.entropy_future)
           self.past = numpy.zeros_like(self.future)
           self.past_entropy = numpy.zeros_like(self.present_entropy)
           self.future_audio = data.copy()
           self.present_audio = numpy.zeros(4096)
           self.clearflag = 0
           self.initialized = 1
           self.stft.reset()
           return data

        else:
           self.past = self.present.copy()
           self.past_entropy = self.present_entropy.copy()
           self.present = self.future.copy()
           self.present_entropy = self.entropy_future.copy()
           self.future = numpy.abs(self.oneshot_hann.analysis(data))
           logit = numpy.abs(self.oneshot_logit.analysis(data))
           self.entropy_future = fast_entropy(numpy.sort(logit[:,0:self.NBINS],axis=1),self.true_logistic)     
           self.present_audio = self.future_audio.copy()
           self.future_audio = data.copy()      
           self.stft.analysis(self.present_audio)
           mask = mask_generation(numpy.vstack((self.past,self.present,self.past)),numpy.hstack((self.past_entropy,self.present_entropy,self.entropy_future)),self.NBINS)
           output = self.stft.synthesis(self.stft.X* mask)
           return output
  
    def stop(self):
        self.running = False 

class StreamSampler(object):

    def __init__(self, sample_rate=48000, channels=2,  micindex=1, speakerindex=1, dtype=numpy.float32):
        self.pa = pyaudio.PyAudio()
        self.processing_size = 4096
        self.sample_rate = sample_rate
        self.channels = channels
        self.rightclearflag = 1
        self.leftclearflag = 1
        self.rightthread = FilterThread(self.rightclearflag)
        self.leftthread = FilterThread(self.leftclearflag)
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
        print("processing!")
        audio_out = audio_in.copy()
        audio_out[:,0] = self.rightthread.process(audio_out[:,0])
        audio_out[:,1] = self.rightthread.process(audio_out[:,1])
        self.speakerstream.write(numpy.column_stack(audio_out).astype(numpy.float32).tobytes())
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
