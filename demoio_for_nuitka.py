'''
FILE IO Demo v2.0
Copyright 2022 Oscar Steila, Joshuah Rainstar
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
'''
Copyright 2022 Oscar Steila, Joshuah Rainstar
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

#How to use this file:
#step one: using https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Windows-x86_64.exe
#install python.
#step three: locate the dedicated python terminal in your start menu, called mambaforge prompt.
#within that prompt, give the following instructions:
#conda update --all
#conda install pip numpy scipy
#pip install tk pyroomacoustics Nuitka ordered-set
#if all of these steps successfully complete, you're ready to go, otherwise fix things.
#run it by executing this command in the terminal, in the folder containing the file: python demo_fileIO.py
#compile it by executing: python -m --standalone --include-plugin=tk-inter demoio_for_nuitka.py

from scipy.io import wavfile
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import time

import os
import numpy
import pyroomacoustics as pra
from threading import Thread

def man(arr):
    med = numpy.nanmedian(arr[numpy.nonzero(arr)])
    return numpy.nanmedian(numpy.abs(arr - med))

def atd(arr):
    x = numpy.square(numpy.abs(arr - man(arr)))
    return numpy.sqrt(numpy.nanmean(x))


def threshold(data: numpy.ndarray):
 a = numpy.sqrt(numpy.nanmean(numpy.square(numpy.abs(data -numpy.nanmedian(numpy.abs(data - numpy.nanmedian(data[numpy.nonzero(data)]))))))) + numpy.nanmedian(data[numpy.nonzero(data)])
 return a

def threshhold(arr):
  return (atd(arr)+ numpy.nanmedian(arr[numpy.nonzero(arr)])) 

def moving_average(x, w):
    return numpy.convolve(x, numpy.ones(w), 'same') / w

def smoothpadded(data: numpy.ndarray,n:float):
  o = numpy.pad(data, n*2, mode='median')
  return moving_average(o,n)[n*2: -n*2]

def numpy_convolve_filter_longways(data: numpy.ndarray,N:int,M:int):
  E = N*2
  d = numpy.pad(array=data,pad_width=((E,E),(0,0)),mode="constant")  
  for each in range(d.shape[0]):
      for all in range(M):
       d[each,:] = (d[each,:]  + (numpy.convolve(d[each,:], numpy.ones(N),mode="same") / N)[:])/2
  return d[E:-E,:]

def numpy_convolve_filter_topways(data: numpy.ndarray,N:int,M:int):
  E = N*2
  d = numpy.pad(array=data,pad_width=((0,0),(E,E)),mode="constant")  
  for each in range(d.shape[1]):
      for all in range(M):
       d[:,each] = (d[:,each]  + (numpy.convolve(d[:,each], numpy.ones(N),mode="same") / N)[:])/2
  return d[:,E:-E]

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


def fast_entropy(data: numpy.ndarray):
   logit = numpy.asarray([0.,0.08507164,0.17014328,0.22147297,0.25905871,0.28917305,0.31461489,0.33688201,0.35687314,0.37517276,0.39218487,0.40820283,0.42344877,0.43809738,0.45229105,0.46614996,0.47977928,0.49327447,0.50672553,0.52022072,0.53385004,0.54770895,0.56190262,0.57655123,0.59179717,0.60781513,0.62482724,0.64312686,0.66311799,0.68538511,0.71082695,0.74094129,0.77852703,0.82985672,0.91492836,1.])
   #note: if you alter the number of bins, you need to regenerate this array. currently set to consider 36 bins
   entropy = numpy.zeros(data.shape[1],dtype=numpy.float32)
   for each in range(data.shape[1]):
      d = data[:,each]
      d = numpy.interp(d, (d[0], d[-1]), (0, +1))
      entropy[each] = 1 - numpy.corrcoef(d, logit)[0,1]
   return entropy

def fast_peaks(stft_:numpy.ndarray,entropy:numpy.ndarray,thresh:numpy.float32,entropy_unmasked:numpy.ndarray):
    #0.01811 practical lowest - but the absolute lowest is 0. 0 is a perfect logistic. 
    #0.595844362 practical highest - an array of zeros with 1 value of 1.
    #note: this calculation is dependent on the size of the logistic array.
    #if you change the number of bins, this constraint will no longer hold and renormalizing will fail.
    #a lookup table is not provided, however the value can be derived by setting up a one-shot fast_entropy that only considers one array,
    #sorting, interpolating, and comparing it. 
    mask = numpy.zeros_like(stft_)
    for each in range(stft_.shape[1]):
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

def mask_generate(stft_vh: numpy.ndarray,stft_vl: numpy.ndarray):

    #24000/256 = 93.75 hz per frequency bin.
    #a 4000 hz window(the largest for SSB is roughly 43 bins.
    #https://en.wikipedia.org/wiki/Voice_frequency
    #however, practically speaking, voice frequency cuts off just above 3400hz.
    #*most* SSB channels are constrained far below this. let's just go with 36 bins.
    #we automatically set all other bins to the residue value.
    #reconstruction or upsampling of this reduced bandwidth signal is a different problem we dont solve here.

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
    
    mask = numpy_convolve_filter_longways(mask,5,17)
    mask2 = numpy_convolve_filter_topways(mask,5,2)
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
           X = self.stft.analysis(data) #generate our complex representation
           Y = self.stftl.analysis(data)
           mask = mask_generate(numpy.abs(X.T),numpy.abs(Y.T))
           output = self.stft.synthesis(X* mask)
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
    return numpy.concatenate((processed), axis=0).astype(dtype=data.dtype)           


if __name__ == '__main__':
# look for input file.wav
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename(filetypes=(('wav files', '*.wav'),), title='Select input file.wav')
    file_path, file_tail = os.path.split(filename)
    infile = filename
    outfile = f"{file_path}/cleanup, {file_tail}"
    print(infile)
    print(outfile)
    rate, data = wavfile.read(infile)
    data = numpy.asarray(data)#correct the format for processing
    reduced_noise = process_data(data)
    wavfile.write(outfile, rate, reduced_noise)
