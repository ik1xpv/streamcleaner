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
#Cleanup.py 1.1 1/15/2023 - OLA stft, non-normalized, double-masked,window-consistent maximum energy filtration method.

#additional code contributed by Justin Engel
#idea and code and bugs by Joshuah Rainstar, Oscar Steila
#https://groups.io/g/NextGenSDRs/topic/spectral_denoising
#discrete.py- discrete frame processing, with overlap
#a "realtime" 140ms latency version is available, with similar performance but inferior filtration.
#considering 1 second at a time allows us the maximum benefit in voice processing.



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
#pip install pipwin dearpygui pyroomacoustics numba
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
#I recommend the use of a notch filter to reduce false positives and carrier signals disrupting the entropy filter,
#albiet when receiving multi-carrier transmitted signal. In those situations turn the notch filter off.
#Please put other denoising methods except for the notch filter *after* this method.
#the use of 1e-6 as a residual multiplier is arbitrary. 

#this program can be somewhat quickly compiled with nuitka provided that all of the ssqueezepy calls are removed and all the numba decorations
#are removed and the numba.prange calls are changed to range and the dearpygui code is removed.
#however, no performance benefits will be realized, as all of the critical logic called from this program is pre-compiled into C and 
#all of the logic which is compute intensive in this program is compiled with nuitka.


import os
import numba
import numpy
import pyroomacoustics as pra
from threading import Thread

import pyaudio
import dearpygui.dearpygui as dpg



@numba.njit(numba.float32(numba.float32[:]))
def man(arr):
    med = numpy.nanmedian(arr[numpy.nonzero(arr)])
    return numpy.nanmedian(numpy.abs(arr - med))

@numba.njit(numba.float32(numba.float32[:]))
def atd(arr):
    x = numpy.square(numpy.abs(arr - man(arr)))
    return numpy.sqrt(numpy.nanmean(x))

@numba.njit(numba.float32(numba.float32[:]))
def threshold(data: numpy.ndarray):
 a = numpy.sqrt(numpy.nanmean(numpy.square(numpy.abs(data -numpy.nanmedian(numpy.abs(data - numpy.nanmedian(data[numpy.nonzero(data)]))))))) + numpy.nanmedian(data[numpy.nonzero(data)])
 return a

def moving_average(x, w):
    return numpy.convolve(x, numpy.ones(w), 'same') / w

def smoothpadded(data: numpy.ndarray,n:float):
  o = numpy.pad(data, n*2, mode='median')
  return moving_average(o,n)[n*2: -n*2]

#there's some marginal error involved here, but the speedup is double
def numpy_convolve_filter_longways1d(data: numpy.ndarray,N:int,M:int):
  E = N*2
  d = numpy.pad(array=data,pad_width=E,mode="constant")  
  b = numpy.ravel(d)
  for all in range(M):
    b[:] = (b[:]  + (numpy.convolve(b[:], numpy.ones(N),mode="same") / N)[:])/2
  return d[E:-E,E:-E]

#there's some marginal error involved here, but the speedup is double
def numpy_convolve_filter_topways1d(data: numpy.ndarray,N:int,M:int):
  E = N*2
  d = numpy.pad(array=data,pad_width=E,mode="constant")  
  d = d.T.copy()
  b = numpy.ravel(d)
  for all in range(M):
    b[:] = (b[:]  + (numpy.convolve(b[:], numpy.ones(N),mode="same") / N)[:])/2
  return d[E:-E,E:-E].T

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

@numba.jit()
def generate_hann(M, sym=True):
    a = [0.5, 0.5]
    fac = numpy.linspace(-numpy.pi, numpy.pi, M)
    w = numpy.zeros(M)
    for k in range(len(a)):
        w += a[k] * numpy.cos(k * fac)
    return w


@numba.njit(numba.float32[:](numba.float32[:,:]))
def fast_entropy(data: numpy.ndarray):
   logit = numpy.asarray([0.,0.08507164,0.17014328,0.22147297,0.25905871,0.28917305,0.31461489,0.33688201,0.35687314,0.37517276,0.39218487,0.40820283,0.42344877,0.43809738,0.45229105,0.46614996,0.47977928,0.49327447,0.50672553,0.52022072,0.53385004,0.54770895,0.56190262,0.57655123,0.59179717,0.60781513,0.62482724,0.64312686,0.66311799,0.68538511,0.71082695,0.74094129,0.77852703,0.82985672,0.91492836,1.])
   #note: if you alter the number of bins, you need to regenerate this array. currently set to consider 36 bins
   entropy = numpy.zeros(data.shape[1],dtype=numpy.float32)
   for each in numba.prange(data.shape[1]):
      d = data[:,each]
      d = numpy.interp(d, (d[0], d[-1]), (0, +1))
      entropy[each] = 1 - numpy.corrcoef(d, logit)[0,1]
   return entropy


@numba.jit(numba.float32[:,:](numba.float32[:,:],numba.int32[:],numba.float32,numba.float32[:]))
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
            mask[0:36,each] =  0
            continue #skip the calculations for this row, it's masked already
        constant = atd(data) + man(data)  #by inlining the calls higher in the function, it only ever sees arrays of one size and shape, which optimizes the code
        test = entropy_unmasked[each]  / 0.6091672572096941 #currently set for 36 bins
        test = abs(test - 1) 
        thresh1 = (thresh*test)
        if numpy.isnan(thresh1):
            thresh1 = constant #catch errors
        constant = (thresh1+constant)/2
        data[data<constant] = 0
        data[data>0] = 1
        mask[0:36,each] = data[:]
    return mask

@numba.njit(numba.float32(numba.float32[:]))
def threshhold(arr):
  return (atd(arr)+ numpy.nanmedian(arr[numpy.nonzero(arr)])) 

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
def mask_generate(stft_vh1: numpy.ndarray,stft_vl1: numpy.ndarray):

    #24000/256 = 93.75 hz per frequency bin.
    #a 4000 hz window(the largest for SSB is roughly 43 bins.
    #https://en.wikipedia.org/wiki/Voice_frequency
    #however, practically speaking, voice frequency cuts off just above 3400hz.
    #*most* SSB channels are constrained far below this. let's just go with 36 bins.
    #we automatically set all other bins to the residue value.
    #reconstruction or upsampling of this reduced bandwidth signal is a different problem we dont solve here.
    stft_vh = numpy.ndarray(shape=stft_vh1.shape, dtype=numpy.float32, order='C') 
    stft_vl = numpy.ndarray(shape=stft_vh1.shape, dtype=numpy.float32,order='C') 
    stft_vh[:] = copy.deepcopy(stft_vh1)
    stft_vl[:] = copy.deepcopy(stft_vl1)#the deepcopy is needed for numba only, because numba doesnt like fortran-order.

    lettuce_euler_macaroni = 0.057 #was grossman constant but that was arbitrarily chosen
    stft_vs = numpy.sort(stft_vl[0:36,:],axis=0) #sort the array
    
    entropy_unmasked = fast_entropy(stft_vs)
    entropy = smoothpadded(entropy_unmasked,14)
    factor = numpy.max(entropy)
    
    if factor < lettuce_euler_macaroni: 
      return stft_vh.T * 1e-6

    entropy[entropy<lettuce_euler_macaroni] = 0
    entropy[entropy>0] = 1
    entropy = entropy.astype(dtype=numpy.int32)
    nbins = numpy.sum(entropy)
    maxstreak = longestConsecutive(entropy)

    #ok, so the shortest vowels are still over 100ms long. That's around 37.59 samples. Call it 38.
    #half of 38 is, go figure, 17.
    #now we do have an issue- what if a vowel is being pronounced in the middle of transition?
    #this *could* result in the end or beginning of words being truncated, but with this criteria,
    #we reasonably establish that there are no regions as long as half a vowel.
    #if there's really messed up speech(hence small segments) but enough of it(hence 22 bins)
    #then we can consider the frame to consist of speech
    # an ionosound sweep is also around or better than 24 samples, also
    # number of bins and streak doubled for hop size 64
    # for hop size = 128, nbins = 22, maxstreak = 16
    #if hop size 256 was chosen, nbins would be 11 and maxstreak would be 11.
    #however, fs/hop = maximum alias-free frequency. For hop size of 64, it's over 300hz.
    if nbins<22 and maxstreak<16:
      return stft_vh.T * 1e-6
    mask=numpy.zeros_like(stft_vh)

    stft_vh1 = stft_vh[0:36,:]
    thresh = threshold(stft_vh1[stft_vh1>man(stft_vl[0:36,:].flatten())])/2

    mask[0:36,:] = fast_peaks(stft_vh[0:36,:],entropy,thresh,entropy_unmasked)
    mask = numpy_convolve_filter_longways1d(mask,5,17)
    mask2 = numpy_convolve_filter_topways1d(mask,5,1)
    mask2 = numpy.where(mask==0,0,mask2)
    mask2[mask2<1e-6] = 1e-6
    return mask2.T


class FilterThread(Thread):
    def __init__(self):
        super(FilterThread, self).__init__()
        self.running = True
        self.NFFT = 512 
        self.NBINS=32
        self.hop = 128
        self.hann = generate_hann(512)
        self.logit = generate_logit_window(512)
        self.synthesis = pra.transform.stft.compute_synthesis_window(self.hann, self.hop)
        self.stft = pra.transform.STFT(N=512, hop=self.hop, analysis_window=self.hann,synthesis_window=self.synthesis ,online=True)
        self.stftl = pra.transform.STFT(N=512, hop=self.hop, analysis_window=self.logit,synthesis_window=self.synthesis ,online=True)

    def process(self,data):         
           self.stft.analysis(data) #generate our complex representation
           self.stftl.analysis(data)
           mask = mask_generate(numpy.abs(self.stft.X.T),numpy.abs(self.stftl.X.T))
           output = self.stft.synthesis(X=self.stft.X* mask)
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
    for each in chunks(data, rate):
        if each.size == rate:
            processed.append(filter.process(each))
        else:
            psize = each.size
            working = padarray(each, rate)
            processed.append(filter.process(working)[0:psize])
    end = time.time()
    print("took ", end - start, " to process ", data.size / rate)
    return numpy.concatenate((processed), axis=0)           
        

class StreamSampler(object):

    def __init__(self, sample_rate=48000, channels=2,  micindex=1, speakerindex=1, dtype=numpy.float32):
        self.pa = pyaudio.PyAudio()
        self.processing_size = 48000
        self.sample_rate = sample_rate
        self.channels = channels
        self.rightclearflag = 1
        self.leftclearflag = 1
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
        audio_out = audio_in.copy()
        chans = []
        chans.append(self.rightthread.process(audio_out[:,0]))
        chans.append(self.leftthread.process(audio_out[:,1]))

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
