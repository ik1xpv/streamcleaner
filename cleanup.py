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
#some tradeoffs had to be considered here- the edge behavior pads with median.
#furthermore, the effort to find the value for the edge of the logistic function was replaced by the standard
#odd-reflect padded value of twice the last minus the next to last. This works much more accurately.
#the thresholding for previous versions considered the entire window. This no longer does, only considering the part
#of the window which is to be processed when determining thresholds. This is to eliminate inconsistent behavior caused by 
#the excessive presence of "ocean floor" created by previous windowing of input.


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
#pip install pipwin dearpygui pyroomacoustics numba ssqueezepy
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
from ssqueezpy import stft as sstft
import pyaudio
import dearpygui.dearpygui as dpg
os.environ['SSQ_PARALLEL'] = '0'



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

def numpy_convolve_filter_topways(data: numpy.ndarray,N:int):
   d = numpy.pad(array=data,pad_width=((N,N),(N,N)),mode="reflect")
   normal = d.T.copy()
   normal_raveled = normal.ravel()
   normal_raveled[:] =  (numpy.convolve(normal_raveled, numpy.ones(N)) / N)[:-N+1]
   e = normal.T
   return e[N:-N,N:-N]

def numpy_convolve_filter_longways(data: numpy.ndarray,N:int):
   d = numpy.pad(array=data,pad_width=((N,N),(N,N)),mode="reflect")
   normal = d.copy()
   normal_raveled = normal.ravel()
   normal_raveled[:] =  (numpy.convolve(normal_raveled, numpy.ones(N)) / N)[:-N+1]
   e = (normal + d)/2
   return e[N:-N,N:-N]

def numpyfilter_wrapper_50_n_topways(data: numpy.ndarray,n:float,iterations: int):
  d = data.copy()
  for i in range(iterations):
    d = d - numpy_convolve_filter_topways(d,n)
  return d
def numpyfilter_wrapper_50_n_longways(data: numpy.ndarray,n:float,iterations: int):
  d = data.copy()
  for i in range(iterations):
    d = numpy_convolve_filter_longways(d,n)
  return d


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
    


hann = numpy.asarray([0.00000000e+00,3.77965773e-05,1.51180595e-04,3.40134910e-04,6.04630957e-04,9.44628746e-04,1.36007687e-03,1.85091253e-03,2.41706151e-03,3.05843822e-03,3.77494569e-03,4.56647559e-03,5.43290826e-03,6.37411270e-03,7.38994662e-03,8.48025644e-03,9.64487731e-03,1.08836332e-02,1.21963367e-02,1.35827895e-02,1.50427819e-02,1.65760932e-02,1.81824916e-02,1.98617342e-02,2.16135671e-02,2.34377255e-02,2.53339336e-02,2.73019047e-02,2.93413412e-02,3.14519350e-02,3.36333667e-02,3.58853068e-02,3.82074146e-02,4.05993391e-02,4.30607187e-02,4.55911813e-02,4.81903443e-02,5.08578147e-02,5.35931893e-02,5.63960544e-02,5.92659864e-02,6.22025514e-02,6.52053053e-02,6.82737943e-02,7.14075543e-02,7.46061116e-02,7.78689827e-02,8.11956742e-02,8.45856832e-02,8.80384971e-02,9.15535940e-02,9.51304424e-02,9.87685015e-02,1.02467221e-01,1.06226043e-01,1.10044397e-01,1.13921708e-01,1.17857388e-01,1.21850843e-01,1.25901469e-01,1.30008654e-01,1.34171776e-01,1.38390206e-01,1.42663307e-01,1.46990432e-01,1.51370928e-01,1.55804131e-01,1.60289372e-01,1.64825973e-01,1.69413247e-01,1.74050502e-01,1.78737036e-01,1.83472140e-01,1.88255099e-01,1.93085190e-01,1.97961681e-01,2.02883837e-01,2.07850913e-01,2.12862158e-01,2.17916814e-01,2.23014117e-01,2.28153297e-01,2.33333576e-01,2.38554171e-01,2.43814294e-01,2.49113148e-01,2.54449933e-01,2.59823842e-01,2.65234062e-01,2.70679775e-01,2.76160159e-01,2.81674384e-01,2.87221617e-01,2.92801019e-01,2.98411747e-01,3.04052952e-01,3.09723782e-01,3.15423378e-01,3.21150881e-01,3.26905422e-01,3.32686134e-01,3.38492141e-01,3.44322565e-01,3.50176526e-01,3.56053138e-01,3.61951513e-01,3.67870760e-01,3.73809982e-01,3.79768282e-01,3.85744760e-01,3.91738511e-01,3.97748631e-01,4.03774209e-01,4.09814335e-01,4.15868096e-01,4.21934577e-01,4.28012860e-01,4.34102027e-01,4.40201156e-01,4.46309327e-01,4.52425614e-01,4.58549094e-01,4.64678841e-01,4.70813928e-01,4.76953428e-01,4.83096412e-01,4.89241951e-01,4.95389117e-01,5.01536980e-01,5.07684611e-01,5.13831080e-01,5.19975458e-01,5.26116815e-01,5.32254225e-01,5.38386758e-01,5.44513487e-01,5.50633486e-01,5.56745831e-01,5.62849596e-01,5.68943859e-01,5.75027699e-01,5.81100196e-01,5.87160431e-01,5.93207489e-01,5.99240456e-01,6.05258418e-01,6.11260467e-01,6.17245695e-01,6.23213197e-01,6.29162070e-01,6.35091417e-01,6.41000339e-01,6.46887944e-01,6.52753341e-01,6.58595644e-01,6.64413970e-01,6.70207439e-01,6.75975174e-01,6.81716305e-01,6.87429962e-01,6.93115283e-01,6.98771407e-01,7.04397480e-01,7.09992651e-01,7.15556073e-01,7.21086907e-01,7.26584315e-01,7.32047467e-01,7.37475536e-01,7.42867702e-01,7.48223150e-01,7.53541070e-01,7.58820659e-01,7.64061117e-01,7.69261652e-01,7.74421479e-01,7.79539817e-01,7.84615893e-01,7.89648938e-01,7.94638193e-01,7.99582902e-01,8.04482319e-01,8.09335702e-01,8.14142317e-01,8.18901439e-01,8.23612347e-01,8.28274329e-01,8.32886681e-01,8.37448705e-01,8.41959711e-01,8.46419017e-01,8.50825950e-01,8.55179843e-01,8.59480037e-01,8.63725883e-01,8.67916738e-01,8.72051970e-01,8.76130952e-01,8.80153069e-01,8.84117711e-01,8.88024281e-01,8.91872186e-01,8.95660845e-01,8.99389686e-01,9.03058145e-01,9.06665667e-01,9.10211707e-01,9.13695728e-01,9.17117204e-01,9.20475618e-01,9.23770461e-01,9.27001237e-01,9.30167455e-01,9.33268638e-01,9.36304317e-01,9.39274033e-01,9.42177336e-01,9.45013788e-01,9.47782960e-01,9.50484434e-01,9.53117800e-01,9.55682662e-01,9.58178630e-01,9.60605328e-01,9.62962389e-01,9.65249456e-01,9.67466184e-01,9.69612237e-01,9.71687291e-01,9.73691033e-01,9.75623159e-01,9.77483377e-01,9.79271407e-01,9.80986977e-01,9.82629829e-01,9.84199713e-01,9.85696393e-01,9.87119643e-01,9.88469246e-01,9.89745000e-01,9.90946711e-01,9.92074198e-01,9.93127290e-01,9.94105827e-01,9.95009663e-01,9.95838660e-01,9.96592693e-01,9.97271648e-01,9.97875422e-01,9.98403924e-01,9.98857075e-01,9.99234805e-01,9.99537058e-01,9.99763787e-01,9.99914959e-01,9.99990551e-01,9.99990551e-01,9.99914959e-01,9.99763787e-01,9.99537058e-01,9.99234805e-01,9.98857075e-01,9.98403924e-01,9.97875422e-01,9.97271648e-01,9.96592693e-01,9.95838660e-01,9.95009663e-01,9.94105827e-01,9.93127290e-01,9.92074198e-01,9.90946711e-01,9.89745000e-01,9.88469246e-01,9.87119643e-01,9.85696393e-01,9.84199713e-01,9.82629829e-01,9.80986977e-01,9.79271407e-01,9.77483377e-01,9.75623159e-01,9.73691033e-01,9.71687291e-01,9.69612237e-01,9.67466184e-01,9.65249456e-01,9.62962389e-01,9.60605328e-01,9.58178630e-01,9.55682662e-01,9.53117800e-01,9.50484434e-01,9.47782960e-01,9.45013788e-01,9.42177336e-01,9.39274033e-01,9.36304317e-01,9.33268638e-01,9.30167455e-01,9.27001237e-01,9.23770461e-01,9.20475618e-01,9.17117204e-01,9.13695728e-01,9.10211707e-01,9.06665667e-01,9.03058145e-01,8.99389686e-01,8.95660845e-01,8.91872186e-01,8.88024281e-01,8.84117711e-01,8.80153069e-01,8.76130952e-01,8.72051970e-01,8.67916738e-01,8.63725883e-01,8.59480037e-01,8.55179843e-01,8.50825950e-01,8.46419017e-01,8.41959711e-01,8.37448705e-01,8.32886681e-01,8.28274329e-01,8.23612347e-01,8.18901439e-01,8.14142317e-01,8.09335702e-01,8.04482319e-01,7.99582902e-01,7.94638193e-01,7.89648938e-01,7.84615893e-01,7.79539817e-01,7.74421479e-01,7.69261652e-01,7.64061117e-01,7.58820659e-01,7.53541070e-01,7.48223150e-01,7.42867702e-01,7.37475536e-01,7.32047467e-01,7.26584315e-01,7.21086907e-01,7.15556073e-01,7.09992651e-01,7.04397480e-01,6.98771407e-01,6.93115283e-01,6.87429962e-01,6.81716305e-01,6.75975174e-01,6.70207439e-01,6.64413970e-01,6.58595644e-01,6.52753341e-01,6.46887944e-01,6.41000339e-01,6.35091417e-01,6.29162070e-01,6.23213197e-01,6.17245695e-01,6.11260467e-01,6.05258418e-01,5.99240456e-01,5.93207489e-01,5.87160431e-01,5.81100196e-01,5.75027699e-01,5.68943859e-01,5.62849596e-01,5.56745831e-01,5.50633486e-01,5.44513487e-01,5.38386758e-01,5.32254225e-01,5.26116815e-01,5.19975458e-01,5.13831080e-01,5.07684611e-01,5.01536980e-01,4.95389117e-01,4.89241951e-01,4.83096412e-01,4.76953428e-01,4.70813928e-01,4.64678841e-01,4.58549094e-01,4.52425614e-01,4.46309327e-01,4.40201156e-01,4.34102027e-01,4.28012860e-01,4.21934577e-01,4.15868096e-01,4.09814335e-01,4.03774209e-01,3.97748631e-01,3.91738511e-01,3.85744760e-01,3.79768282e-01,3.73809982e-01,3.67870760e-01,3.61951513e-01,3.56053138e-01,3.50176526e-01,3.44322565e-01,3.38492141e-01,3.32686134e-01,3.26905422e-01,3.21150881e-01,3.15423378e-01,3.09723782e-01,3.04052952e-01,2.98411747e-01,2.92801019e-01,2.87221617e-01,2.81674384e-01,2.76160159e-01,2.70679775e-01,2.65234062e-01,2.59823842e-01,2.54449933e-01,2.49113148e-01,2.43814294e-01,2.38554171e-01,2.33333576e-01,2.28153297e-01,2.23014117e-01,2.17916814e-01,2.12862158e-01,2.07850913e-01,2.02883837e-01,1.97961681e-01,1.93085190e-01,1.88255099e-01,1.83472140e-01,1.78737036e-01,1.74050502e-01,1.69413247e-01,1.64825973e-01,1.60289372e-01,1.55804131e-01,1.51370928e-01,1.46990432e-01,1.42663307e-01,1.38390206e-01,1.34171776e-01,1.30008654e-01,1.25901469e-01,1.21850843e-01,1.17857388e-01,1.13921708e-01,1.10044397e-01,1.06226043e-01,1.02467221e-01,9.87685015e-02,9.51304424e-02,9.15535940e-02,8.80384971e-02,8.45856832e-02,8.11956742e-02,7.78689827e-02,7.46061116e-02,7.14075543e-02,6.82737943e-02,6.52053053e-02,6.22025514e-02,5.92659864e-02,5.63960544e-02,5.35931893e-02,5.08578147e-02,4.81903443e-02,4.55911813e-02,4.30607187e-02,4.05993391e-02,3.82074146e-02,3.58853068e-02,3.36333667e-02,3.14519350e-02,2.93413412e-02,2.73019047e-02,2.53339336e-02,2.34377255e-02,2.16135671e-02,1.98617342e-02,1.81824916e-02,1.65760932e-02,1.50427819e-02,1.35827895e-02,1.21963367e-02,1.08836332e-02,9.64487731e-03,8.48025644e-03,7.38994662e-03,6.37411270e-03,5.43290826e-03,4.56647559e-03,3.77494569e-03,3.05843822e-03,2.41706151e-03,1.85091253e-03,1.36007687e-03,9.44628746e-04,6.04630957e-04,3.40134910e-04,1.51180595e-04,3.77965773e-05,0.00000000e+00])
logit_window = numpy.asarray([0.,0.05590667,0.11181333,0.14464919,0.16804013,0.18625637,0.20119997,0.21388557,0.22491881,0.23469034,0.24346692,0.2514388,0.25874646,0.26549659,0.27177213,0.27763882,0.28314967,0.28834802,0.2932698,0.29794509,0.30239936,0.30665433,0.31072869,0.31463866,0.31839837,0.32202023,0.32551518,0.32889293,0.33216214,0.33533054,0.3384051,0.34139207,0.34429715,0.34712548,0.34988176,0.35257028,0.35519495,0.35775939,0.36026692,0.36272059,0.36512323,0.36747746,0.36978573,0.37205028,0.37427323,0.37645655,0.37860207,0.38071152,0.3827865,0.38482855,0.38683907,0.38881941,0.39077084,0.39269455,0.39459167,0.39646327,0.39831036,0.40013389,0.40193478,0.4037139,0.40547206,0.40721004,0.4089286,0.41062845,0.41231027,0.4139747,0.41562237,0.41725387,0.41886977,0.42047061,0.42205693,0.42362923,0.42518798,0.42673365,0.4282667,0.42978755,0.43129661,0.43279429,0.43428097,0.43575703,0.43722282,0.43867871,0.44012502,0.44156207,0.4429902,0.44440971,0.44582088,0.44722402,0.44861941,0.45000731,0.451388,0.45276174,0.45412877,0.45548934,0.4568437,0.45819207,0.45953469,0.46087178,0.46220355,0.46353023,0.46485202,0.46616913,0.46748176,0.46879011,0.47009437,0.47139473,0.47269138,0.4739845,0.47527428,0.4765609,0.47784452,0.47912534,0.48040351,0.48167921,0.48295261,0.48422387,0.48549315,0.48676063,0.48802646,0.4892908,0.49055382,0.49181567,0.4930765,0.49433648,0.49559576,0.4968545,0.49811286,0.49937098,0.50062902,0.50188714,0.5031455,0.50440424,0.50566352,0.5069235,0.50818433,0.50944618,0.5107092,0.51197354,0.51323937,0.51450685,0.51577613,0.51704739,0.51832079,0.51959649,0.52087466,0.52215548,0.5234391,0.52472572,0.5260155,0.52730862,0.52860527,0.52990563,0.53120989,0.53251824,0.53383087,0.53514798,0.53646977,0.53779645,0.53912822,0.54046531,0.54180793,0.5431563,0.54451066,0.54587123,0.54723826,0.548612,0.54999269,0.55138059,0.55277598,0.55417912,0.55559029,0.5570098,0.55843793,0.55987498,0.56132129,0.56277718,0.56424297,0.56571903,0.56720571,0.56870339,0.57021245,0.5717333,0.57326635,0.57481202,0.57637077,0.57794307,0.57952939,0.58113023,0.58274613,0.58437763,0.5860253,0.58768973,0.58937155,0.5910714,0.59278996,0.59452794,0.5962861,0.59806522,0.59986611,0.60168964,0.60353673,0.60540833,0.60730545,0.60922916,0.61118059,0.61316093,0.61517145,0.6172135,0.61928848,0.62139793,0.62354345,0.62572677,0.62794972,0.63021427,0.63252254,0.63487677,0.63727941,0.63973308,0.64224061,0.64480505,0.64742972,0.65011824,0.65287452,0.65570285,0.65860793,0.6615949,0.66466946,0.66783786,0.67110707,0.67448482,0.67797977,0.68160163,0.68536134,0.68927131,0.69334567,0.69760064,0.70205491,0.7067302,0.71165198,0.71685033,0.72236118,0.72822787,0.73450341,0.74125354,0.7485612,0.75653308,0.76530966,0.77508119,0.78611443,0.79880003,0.81374363,0.83195987,0.85535081,0.88818667,0.94409333,1.,1.,0.94409333,0.88818667,0.85535081,0.83195987,0.81374363,0.79880003,0.78611443,0.77508119,0.76530966,0.75653308,0.7485612,0.74125354,0.73450341,0.72822787,0.72236118,0.71685033,0.71165198,0.7067302,0.70205491,0.69760064,0.69334567,0.68927131,0.68536134,0.68160163,0.67797977,0.67448482,0.67110707,0.66783786,0.66466946,0.6615949,0.65860793,0.65570285,0.65287452,0.65011824,0.64742972,0.64480505,0.64224061,0.63973308,0.63727941,0.63487677,0.63252254,0.63021427,0.62794972,0.62572677,0.62354345,0.62139793,0.61928848,0.6172135,0.61517145,0.61316093,0.61118059,0.60922916,0.60730545,0.60540833,0.60353673,0.60168964,0.59986611,0.59806522,0.5962861,0.59452794,0.59278996,0.5910714,0.58937155,0.58768973,0.5860253,0.58437763,0.58274613,0.58113023,0.57952939,0.57794307,0.57637077,0.57481202,0.57326635,0.5717333,0.57021245,0.56870339,0.56720571,0.56571903,0.56424297,0.56277718,0.56132129,0.55987498,0.55843793,0.5570098,0.55559029,0.55417912,0.55277598,0.55138059,0.54999269,0.548612,0.54723826,0.54587123,0.54451066,0.5431563,0.54180793,0.54046531,0.53912822,0.53779645,0.53646977,0.53514798,0.53383087,0.53251824,0.53120989,0.52990563,0.52860527,0.52730862,0.5260155,0.52472572,0.5234391,0.52215548,0.52087466,0.51959649,0.51832079,0.51704739,0.51577613,0.51450685,0.51323937,0.51197354,0.5107092,0.50944618,0.50818433,0.5069235,0.50566352,0.50440424,0.5031455,0.50188714,0.50062902,0.49937098,0.49811286,0.4968545,0.49559576,0.49433648,0.4930765,0.49181567,0.49055382,0.4892908,0.48802646,0.48676063,0.48549315,0.48422387,0.48295261,0.48167921,0.48040351,0.47912534,0.47784452,0.4765609,0.47527428,0.4739845,0.47269138,0.47139473,0.47009437,0.46879011,0.46748176,0.46616913,0.46485202,0.46353023,0.46220355,0.46087178,0.45953469,0.45819207,0.4568437,0.45548934,0.45412877,0.45276174,0.451388,0.45000731,0.44861941,0.44722402,0.44582088,0.44440971,0.4429902,0.44156207,0.44012502,0.43867871,0.43722282,0.43575703,0.43428097,0.43279429,0.43129661,0.42978755,0.4282667,0.42673365,0.42518798,0.42362923,0.42205693,0.42047061,0.41886977,0.41725387,0.41562237,0.4139747,0.41231027,0.41062845,0.4089286,0.40721004,0.40547206,0.4037139,0.40193478,0.40013389,0.39831036,0.39646327,0.39459167,0.39269455,0.39077084,0.38881941,0.38683907,0.38482855,0.3827865,0.38071152,0.37860207,0.37645655,0.37427323,0.37205028,0.36978573,0.36747746,0.36512323,0.36272059,0.36026692,0.35775939,0.35519495,0.35257028,0.34988176,0.34712548,0.34429715,0.34139207,0.3384051,0.33533054,0.33216214,0.32889293,0.32551518,0.32202023,0.31839837,0.31463866,0.31072869,0.30665433,0.30239936,0.29794509,0.2932698,0.28834802,0.28314967,0.27763882,0.27177213,0.26549659,0.25874646,0.2514388,0.24346692,0.23469034,0.22491881,0.21388557,0.20119997,0.18625637,0.16804013,0.14464919,0.11181333,0.05590667,0.])
#predefining our windows also saves some cycles 
#D. Griffin and J. Lim, Signal estimation from modified short-time Fourier transform, IEEE Trans. Acoustics, Speech, and Signal Process., vol. 32, no. 2, pp. 236-243, 1984.
#optimal synthesis window generated with pyroomacoustics using pyroomacoustics.transform.stft.compute_synthesis_window(stftwindow, hop)

def mask_generate(data: numpy.ndarray):

    #24000/256 = 93.75 hz per frequency bin.
    #a 4000 hz window(the largest for SSB is roughly 43 bins.
    #https://en.wikipedia.org/wiki/Voice_frequency
    #however, practically speaking, voice frequency cuts off just above 3400hz.
    #*most* SSB channels are constrained far below this. let's just go with 36 bins.
    #we automatically set all other bins to the residue value.
    #reconstruction or upsampling of this reduced bandwidth signal is a different problem we dont solve here.
 
    data= numpy.asarray(data,dtype=float) #correct byte order of array if it is incorrect
    lettuce_euler_macaroni = 0.057 #was grossman constant but that was arbitrarily chosen

    stft_logit = sstft(x=data,window=logit_window,n_fft=512,hop_len=128)
    stft_vl =  numpy.abs(stft_logit) #returns the same as other methods
    stft_hann = sstft(x=data,window=hann,n_fft=512,hop_len=128) #get complex representation
    stft_vh =  numpy.abs(stft_hann) #returns the same as other methods

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
    mask1=numpy.zeros_like(stft_vh)
    mask2=numpy.zeros_like(stft_vh)        

    stft_vh1 = stft_vh[0:36,:]
    thresh = threshold(stft_vh1[stft_vh1>man(stft_vl[0:36,:].flatten())])/2

    mask1[0:36,:] = fast_peaks(stft_vh[0:36,:],entropy,thresh,entropy_unmasked)
    mask2 = numpyfilter_wrapper_50_n_longways(mask1,3,15)
    mask3 = numpyfilter_wrapper_50_n_topways(mask2*2,31,2)
    mask3 = numpy.where(mask2==0,0,mask3) #dont conserve added energy, this is for attenuation only
    mask2[mask2>1]=1
    mask3[mask3>1]=1

    mask3 = (mask3 + mask2)/2 #converge the results
    mask3[mask3<1e-6] = 1e-6

    return mask3.T



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

        self.residual = 0

    def process(self,data):         
           self.stft.analysis(data) #generate our complex representation
           mask = mask_generate(data)
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
