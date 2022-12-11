'''
Demo v1.1.7
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


#idea and code and bugs by Joshuah Rainstar   : https://groups.io/g/NextGenSDRs/message/1085
#fork, mod by Oscar Steila : https://groups.io/g/NextGenSDRs/topic/spectral_denoising
#cleanup_interactive1.1.7.py

#12/11/2022 : this is no longer an experiment

#instructions: 

#How to use this file:
#you will need 1 virtual audio cable- try https://vb-audio.com/Cable/ if you use windows.
#install and configure the virtual audio cable and your speakers for 16 bits, 48000hz, two channels.
#if you use OSX or Linux, you will have to modify this file by changing audio devices, settings and libraries appropriately.
#step one: put this file somewhere and remember where it is.
#step two: using https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Windows-x86_64.exe
#install python.
#step three: locate the dedicated python terminal in your start menu, called mambaforge prompt.
#within that prompt, give the following instructions:
#pip install numpy
#pip install librosa pipwin dearpygui np-rw-buffer opencv-python numba matplotlib
#pipwin install pyaudio
#if all three these steps successfully complete, you're ready to go, otherwise fix things.
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
import numba
import pyaudio
import librosa
from librosa import stft, istft
from time import sleep
from np_rw_buffer import AudioFramingBuffer
import dearpygui.dearpygui as dpg
import array

import matplotlib.image as mimage
import matplotlib.colors as colors
from matplotlib import cm
import cv2


STFT_DISPLAY_HZ = 8000
NROWS = int((512 * STFT_DISPLAY_HZ)/48000) # spectrogram height 257 => 24 kHz , 85 => 8 kHz
ACTIVE_BAND = 3400
NBAND = int((512 * ACTIVE_BAND)/48000) # rx bandwidth


#@numba.jit()
#def boxcar(M, sym=True):
#    a = [0.5, 0.5]
#    fac = numpy.linspace(-numpy.pi, numpy.pi, M)
#    w = numpy.zeros(M)
#    for k in range(len(a)):
#        w += a[k] * numpy.cos(k * fac)
#        return w

#@numba.jit()
#def hann(M, sym=True):
#    a = [0.5, 0.5]
#    fac = numpy.linspace(-numpy.pi, numpy.pi, M)
#    w = numpy.zeros(M)
#    for k in range(len(a)):
#        w += a[k] * numpy.cos(k * fac)
#    return w

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

def corrected_logit(size):
    fprint = numpy.linspace(0, 1, size)
    fprint [1:-1] /= 1 - fprint [1:-1]
    fprint [1:-1] = numpy.log(fprint [1:-1])
    fprint[0] = -6
    fprint[-1] = 6
    return fprint

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


@numba.jit()
def fast_entropy(data: numpy.ndarray):
   logit = numpy.asarray([-6.,-3.40119738,-2.67414865,-2.23359222,-1.9095425,-1.64865863,-1.42711636,-1.23214368,-1.05605267,-0.89381788,-0.74193734,-0.597837,-0.45953233,-0.3254224,-0.19415601,-0.06453852,0.06453852,0.19415601,0.3254224,0.45953233,0.597837,0.74193734,0.89381788,1.05605267,1.23214368,1.42711636,1.64865863,1.9095425,2.23359222,2.67414865,3.40119738,6.])
    #predefining the logit distribution saves some cycles. This is the same as corrected_logit(32).
   entropy = numpy.zeros(data.shape[1])
   for each in numba.prange(data.shape[1]):
      d = data[:,each]
      d = numpy.interp(d, (d[0], d[-1]), (-6, +6))
      entropy[each] = 1 - numpy.corrcoef(d, logit)[0,1]
   return entropy



@numba.jit()
def fast_peaks(stft_:numpy.ndarray,entropy:numpy.ndarray,thresh:numpy.float64,entropy_unmasked:numpy.ndarray):
    mask = numpy.zeros_like(stft_)
    for each in numba.prange(stft_.shape[1]):
        data = stft_[:,each]
        if entropy[each] == 0:
            mask[0:32,each] =  0
            continue #skip the calculations for this row, it's masked already
        constant = atd(data) + man(data)  #by inlining the calls higher in the function, it only ever sees arrays of one size and shape, which optimizes the code
        if entropy_unmasked[each] >0.055012436840346526:
            test = (entropy_unmasked[each]  - 0.055012436840346526) / (0.20608218909223255  - 0.055012436840346526)
        else:
            test = 0
        test = abs(test - 1) 
        thresh1 = (thresh*test)
        if numpy.isnan(thresh1):
            thresh1 = constant #catch errors
        constant = (thresh1+constant)/2
        data[data<constant] = 0
        data[data>0] = 1
        mask[0:32,each] = data[:]
    return mask

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
    
@numba.jit()
def threshhold(arr):
  return (atd(arr)+ numpy.nanmedian(arr[numpy.nonzero(arr)]))


color_scale = numpy.linspace((0,0,0,0,0),(1,1,1,0,0), num= NROWS) 

boxcar = numpy.asarray([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
hann = numpy.asarray([0.00000000e+00,3.77965773e-05,1.51180595e-04,3.40134910e-04,6.04630957e-04,9.44628746e-04,1.36007687e-03,1.85091253e-03,2.41706151e-03,3.05843822e-03,3.77494569e-03,4.56647559e-03,5.43290826e-03,6.37411270e-03,7.38994662e-03,8.48025644e-03,9.64487731e-03,1.08836332e-02,1.21963367e-02,1.35827895e-02,1.50427819e-02,1.65760932e-02,1.81824916e-02,1.98617342e-02,2.16135671e-02,2.34377255e-02,2.53339336e-02,2.73019047e-02,2.93413412e-02,3.14519350e-02,3.36333667e-02,3.58853068e-02,3.82074146e-02,4.05993391e-02,4.30607187e-02,4.55911813e-02,4.81903443e-02,5.08578147e-02,5.35931893e-02,5.63960544e-02,5.92659864e-02,6.22025514e-02,6.52053053e-02,6.82737943e-02,7.14075543e-02,7.46061116e-02,7.78689827e-02,8.11956742e-02,8.45856832e-02,8.80384971e-02,9.15535940e-02,9.51304424e-02,9.87685015e-02,1.02467221e-01,1.06226043e-01,1.10044397e-01,1.13921708e-01,1.17857388e-01,1.21850843e-01,1.25901469e-01,1.30008654e-01,1.34171776e-01,1.38390206e-01,1.42663307e-01,1.46990432e-01,1.51370928e-01,1.55804131e-01,1.60289372e-01,1.64825973e-01,1.69413247e-01,1.74050502e-01,1.78737036e-01,1.83472140e-01,1.88255099e-01,1.93085190e-01,1.97961681e-01,2.02883837e-01,2.07850913e-01,2.12862158e-01,2.17916814e-01,2.23014117e-01,2.28153297e-01,2.33333576e-01,2.38554171e-01,2.43814294e-01,2.49113148e-01,2.54449933e-01,2.59823842e-01,2.65234062e-01,2.70679775e-01,2.76160159e-01,2.81674384e-01,2.87221617e-01,2.92801019e-01,2.98411747e-01,3.04052952e-01,3.09723782e-01,3.15423378e-01,3.21150881e-01,3.26905422e-01,3.32686134e-01,3.38492141e-01,3.44322565e-01,3.50176526e-01,3.56053138e-01,3.61951513e-01,3.67870760e-01,3.73809982e-01,3.79768282e-01,3.85744760e-01,3.91738511e-01,3.97748631e-01,4.03774209e-01,4.09814335e-01,4.15868096e-01,4.21934577e-01,4.28012860e-01,4.34102027e-01,4.40201156e-01,4.46309327e-01,4.52425614e-01,4.58549094e-01,4.64678841e-01,4.70813928e-01,4.76953428e-01,4.83096412e-01,4.89241951e-01,4.95389117e-01,5.01536980e-01,5.07684611e-01,5.13831080e-01,5.19975458e-01,5.26116815e-01,5.32254225e-01,5.38386758e-01,5.44513487e-01,5.50633486e-01,5.56745831e-01,5.62849596e-01,5.68943859e-01,5.75027699e-01,5.81100196e-01,5.87160431e-01,5.93207489e-01,5.99240456e-01,6.05258418e-01,6.11260467e-01,6.17245695e-01,6.23213197e-01,6.29162070e-01,6.35091417e-01,6.41000339e-01,6.46887944e-01,6.52753341e-01,6.58595644e-01,6.64413970e-01,6.70207439e-01,6.75975174e-01,6.81716305e-01,6.87429962e-01,6.93115283e-01,6.98771407e-01,7.04397480e-01,7.09992651e-01,7.15556073e-01,7.21086907e-01,7.26584315e-01,7.32047467e-01,7.37475536e-01,7.42867702e-01,7.48223150e-01,7.53541070e-01,7.58820659e-01,7.64061117e-01,7.69261652e-01,7.74421479e-01,7.79539817e-01,7.84615893e-01,7.89648938e-01,7.94638193e-01,7.99582902e-01,8.04482319e-01,8.09335702e-01,8.14142317e-01,8.18901439e-01,8.23612347e-01,8.28274329e-01,8.32886681e-01,8.37448705e-01,8.41959711e-01,8.46419017e-01,8.50825950e-01,8.55179843e-01,8.59480037e-01,8.63725883e-01,8.67916738e-01,8.72051970e-01,8.76130952e-01,8.80153069e-01,8.84117711e-01,8.88024281e-01,8.91872186e-01,8.95660845e-01,8.99389686e-01,9.03058145e-01,9.06665667e-01,9.10211707e-01,9.13695728e-01,9.17117204e-01,9.20475618e-01,9.23770461e-01,9.27001237e-01,9.30167455e-01,9.33268638e-01,9.36304317e-01,9.39274033e-01,9.42177336e-01,9.45013788e-01,9.47782960e-01,9.50484434e-01,9.53117800e-01,9.55682662e-01,9.58178630e-01,9.60605328e-01,9.62962389e-01,9.65249456e-01,9.67466184e-01,9.69612237e-01,9.71687291e-01,9.73691033e-01,9.75623159e-01,9.77483377e-01,9.79271407e-01,9.80986977e-01,9.82629829e-01,9.84199713e-01,9.85696393e-01,9.87119643e-01,9.88469246e-01,9.89745000e-01,9.90946711e-01,9.92074198e-01,9.93127290e-01,9.94105827e-01,9.95009663e-01,9.95838660e-01,9.96592693e-01,9.97271648e-01,9.97875422e-01,9.98403924e-01,9.98857075e-01,9.99234805e-01,9.99537058e-01,9.99763787e-01,9.99914959e-01,9.99990551e-01,9.99990551e-01,9.99914959e-01,9.99763787e-01,9.99537058e-01,9.99234805e-01,9.98857075e-01,9.98403924e-01,9.97875422e-01,9.97271648e-01,9.96592693e-01,9.95838660e-01,9.95009663e-01,9.94105827e-01,9.93127290e-01,9.92074198e-01,9.90946711e-01,9.89745000e-01,9.88469246e-01,9.87119643e-01,9.85696393e-01,9.84199713e-01,9.82629829e-01,9.80986977e-01,9.79271407e-01,9.77483377e-01,9.75623159e-01,9.73691033e-01,9.71687291e-01,9.69612237e-01,9.67466184e-01,9.65249456e-01,9.62962389e-01,9.60605328e-01,9.58178630e-01,9.55682662e-01,9.53117800e-01,9.50484434e-01,9.47782960e-01,9.45013788e-01,9.42177336e-01,9.39274033e-01,9.36304317e-01,9.33268638e-01,9.30167455e-01,9.27001237e-01,9.23770461e-01,9.20475618e-01,9.17117204e-01,9.13695728e-01,9.10211707e-01,9.06665667e-01,9.03058145e-01,8.99389686e-01,8.95660845e-01,8.91872186e-01,8.88024281e-01,8.84117711e-01,8.80153069e-01,8.76130952e-01,8.72051970e-01,8.67916738e-01,8.63725883e-01,8.59480037e-01,8.55179843e-01,8.50825950e-01,8.46419017e-01,8.41959711e-01,8.37448705e-01,8.32886681e-01,8.28274329e-01,8.23612347e-01,8.18901439e-01,8.14142317e-01,8.09335702e-01,8.04482319e-01,7.99582902e-01,7.94638193e-01,7.89648938e-01,7.84615893e-01,7.79539817e-01,7.74421479e-01,7.69261652e-01,7.64061117e-01,7.58820659e-01,7.53541070e-01,7.48223150e-01,7.42867702e-01,7.37475536e-01,7.32047467e-01,7.26584315e-01,7.21086907e-01,7.15556073e-01,7.09992651e-01,7.04397480e-01,6.98771407e-01,6.93115283e-01,6.87429962e-01,6.81716305e-01,6.75975174e-01,6.70207439e-01,6.64413970e-01,6.58595644e-01,6.52753341e-01,6.46887944e-01,6.41000339e-01,6.35091417e-01,6.29162070e-01,6.23213197e-01,6.17245695e-01,6.11260467e-01,6.05258418e-01,5.99240456e-01,5.93207489e-01,5.87160431e-01,5.81100196e-01,5.75027699e-01,5.68943859e-01,5.62849596e-01,5.56745831e-01,5.50633486e-01,5.44513487e-01,5.38386758e-01,5.32254225e-01,5.26116815e-01,5.19975458e-01,5.13831080e-01,5.07684611e-01,5.01536980e-01,4.95389117e-01,4.89241951e-01,4.83096412e-01,4.76953428e-01,4.70813928e-01,4.64678841e-01,4.58549094e-01,4.52425614e-01,4.46309327e-01,4.40201156e-01,4.34102027e-01,4.28012860e-01,4.21934577e-01,4.15868096e-01,4.09814335e-01,4.03774209e-01,3.97748631e-01,3.91738511e-01,3.85744760e-01,3.79768282e-01,3.73809982e-01,3.67870760e-01,3.61951513e-01,3.56053138e-01,3.50176526e-01,3.44322565e-01,3.38492141e-01,3.32686134e-01,3.26905422e-01,3.21150881e-01,3.15423378e-01,3.09723782e-01,3.04052952e-01,2.98411747e-01,2.92801019e-01,2.87221617e-01,2.81674384e-01,2.76160159e-01,2.70679775e-01,2.65234062e-01,2.59823842e-01,2.54449933e-01,2.49113148e-01,2.43814294e-01,2.38554171e-01,2.33333576e-01,2.28153297e-01,2.23014117e-01,2.17916814e-01,2.12862158e-01,2.07850913e-01,2.02883837e-01,1.97961681e-01,1.93085190e-01,1.88255099e-01,1.83472140e-01,1.78737036e-01,1.74050502e-01,1.69413247e-01,1.64825973e-01,1.60289372e-01,1.55804131e-01,1.51370928e-01,1.46990432e-01,1.42663307e-01,1.38390206e-01,1.34171776e-01,1.30008654e-01,1.25901469e-01,1.21850843e-01,1.17857388e-01,1.13921708e-01,1.10044397e-01,1.06226043e-01,1.02467221e-01,9.87685015e-02,9.51304424e-02,9.15535940e-02,8.80384971e-02,8.45856832e-02,8.11956742e-02,7.78689827e-02,7.46061116e-02,7.14075543e-02,6.82737943e-02,6.52053053e-02,6.22025514e-02,5.92659864e-02,5.63960544e-02,5.35931893e-02,5.08578147e-02,4.81903443e-02,4.55911813e-02,4.30607187e-02,4.05993391e-02,3.82074146e-02,3.58853068e-02,3.36333667e-02,3.14519350e-02,2.93413412e-02,2.73019047e-02,2.53339336e-02,2.34377255e-02,2.16135671e-02,1.98617342e-02,1.81824916e-02,1.65760932e-02,1.50427819e-02,1.35827895e-02,1.21963367e-02,1.08836332e-02,9.64487731e-03,8.48025644e-03,7.38994662e-03,6.37411270e-03,5.43290826e-03,4.56647559e-03,3.77494569e-03,3.05843822e-03,2.41706151e-03,1.85091253e-03,1.36007687e-03,9.44628746e-04,6.04630957e-04,3.40134910e-04,1.51180595e-04,3.77965773e-05,0.00000000e+00])
#predefining our windows also saves some cycles    


def denoise(data: numpy.ndarray,DENOISE,ENTROPY):

    #24000/256 = 93.75 hz per frequency bin.
    #a 4000 hz window(the largest for SSB is roughly 43 bins.
    #https://en.wikipedia.org/wiki/Voice_frequency
    #however, practically speaking, voice frequency cuts off just above 3400hz.
    #*most* SSB channels are constrained far below this.
    #to catch most voice activity on shortwave, we use the first 32 bins, or 3000hz.
    #we automatically set all other bins to the residue value.
    #reconstruction or upsampling of this reduced bandwidth signal is a different problem we dont solve here.
 
    data= numpy.asarray(data,dtype=float) #correct byte order of array   
    lettuce_euler_macaroni = 0.059 #use the euler/macaroni constant for noise similarity- truncate to 6 points of precision and raise by 1
    #this provides a safe constraint.


    stft_boxcar = stft(data,n_fft=512,window=boxcar) #get complex representation
    stft_vb =  numpy.abs(stft_boxcar) #returns the same as other methods
    stft_vb=(stft_vb-numpy.nanmin(stft_vb))/numpy.ptp(stft_vb) #normalize to 0,1
    floor = threshold(stft_vb.flatten())

    stft_vb  = stft_vb[0:32,:] #obtain the desired bins
    stft_vb = numpy.sort(stft_vb,axis=0) #sort the array
    ent_box = fast_entropy(stft_vb)
    ent_box = ent_box - numpy.min(ent_box )
    ent_box = smoothpadded(ent_box)


    stft_hann = stft(data,n_fft=512,window=hann) #get complex representation
    stft_vh =  numpy.abs(stft_hann) #returns the same as other methods
    stft_vh = (stft_vh -numpy.nanmin(stft_vh))/numpy.ptp(stft_vh) #normalize to 0,1
    
    arr_color = cm.ScalarMappable(cmap="turbo").to_rgba(stft_vr, bytes=False, norm=True) #only the first NROWS
    arr_color = numpy.flipud(arr_color) #updown freq axis
    arr_color = cv2.resize(arr_color, dsize=(660, 257), interpolation=cv2.INTER_CUBIC)
    dpg.set_value("dirty_texture", arr_color) 
    if DENOISE == False: 
      dpg.set_value("clean_texture", arr_color)
      return data
    
    stft_d = stft_vh[stft_vh<floor]
    stft_d = stft_d[stft_d>0]
    residue = man(stft_d)

    stft_vr  = stft_vh[0:32,:].copy() #obtain the desired bins
    stft_vr = numpy.sort(stft_vr,axis=0) #sort the array
    ent_hann = fast_entropy(stft_vr)
    ent_hann = smoothpadded(ent_hann)

    minent = numpy.minimum(ent_hann,ent_box)
    maxent = numpy.maximum(ent_hann,ent_box)
    factor = numpy.max(maxent)

    
    if factor < lettuce_euler_macaroni and ENTROPY == True : #sometimes the old ways are the best ways
      stft_hann = stft_hann * residue
      stft_vh = stft_vh * residue #no point wasting cycles
      processed = istft(stft_hann,window=hann)
      arr_color = cm.ScalarMappable(cmap="turbo").to_rgba(stft_vh, bytes=False, norm=True) #only the first NROWS
      arr_color = numpy.flipud(arr_color) #updown freq axis
      arr_color = cv2.resize(arr_color, dsize=(660, 257), interpolation=cv2.INTER_CUBIC)
      dpg.set_value("clean_texture", arr_color) 
      return processed
      #no point wasting cycles smoothing information which isn't there!

    entropy = (maxent+minent)/2
    entropy_unmasked = entropy.copy()
    entropy[entropy<lettuce_euler_macaroni] = 0
    entropy[entropy>0] = 1
    nbins = numpy.sum(entropy)
    maxstreak = longestConsecutive(entropy)
    
    #ok, so the shortest vowels are still over 100ms long. That's around 37.59 samples. Call it 38.
    #half of 38 is, go figure, 17.
    #now we do have an issue- what if a vowel is being pronounced in the middle of transition?
    #this *could* result in the end or beginning of words being truncated, but with this criteria,
    #we reasonably establish that there are no regions as long as half a vowel.
    #if there's really messed up speech(hence small segments) but enough of it(hence 22 bins)
    #then we can consider the frame to consist of speech
    if nbins<22 and maxstreak<16 and ENTROPY:
      stft_hann = stft_hann  * residue
      processed = istft(stft_hann ,window=hann)
      return processed
          

    mask=numpy.zeros_like(stft_vh)
    thresh = threshhold(numpy.ravel(stft_vh[stft_vh>=floor])) - man(numpy.ravel(stft_vh[stft_vh>=floor])) #we return to being uncertain about this
    
    if ENTROPY:
        mask = fast_peaks(stft_vh[0:32,:],entropy,thresh,entropy_unmasked)
    else:
        mask  = numpy.where(stft_vr>=thresh, 1.0,0)
      
    mask = numpyfilter_wrapper_50(mask)
    mask=(mask-numpy.nanmin(mask))/numpy.ptp(mask)#correct basis    
    mask[mask==0] = residue

    stft_hann = stft_hann * mask
    stft_vh = stft_vh * mask
    arr_color = cm.ScalarMappable(cmap="turbo").to_rgba(stft_vh, bytes=False, norm=True) #only the first NROWS
    arr_color = numpy.flipud(arr_color) #updown freq axis
    arr_color = cv2.resize(arr_color, dsize=(660, 257), interpolation=cv2.INTER_CUBIC)
    dpg.set_value("clean_texture", arr_color) 
    
    processed = istft(stft_hann,window=hann)
    return processed  
  
  
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
        self.denoise = True
        self.entropy = True


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
        chans.append(denoise(audio[:, 0],self.denoise,self.entropy))
        chans.append(denoise(audio[:, 1],self.denoise,self.entropy))
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
        if SS.denoise == True:
            dpg.set_item_label("toggleswitch", "denoiser is OFF")
            SS.denoise = False
        else:
            dpg.set_item_label("toggleswitch", "denoiser is ON ")
            SS.denoise = True
    def entropytoggle(sender, app_data,user_data):
        if SS.entropy == True:
            dpg.set_item_label("toggleswitch2", "Entropy is OFF")
            SS.entropy = False
        else:
            dpg.set_item_label("toggleswitch2", "entropy is ON ")
            SS.entropy = True


    cleantexture = [0.5, 0.5, 0, 1] * 660 * 257
    dirtytexture = [0.5, 0, 0.5, 1] * 660 * 257
    #patch from joviex- the enumeration in the online docs showing .append doesn't work for larger textures        
    with dpg.texture_registry():
        dpg.add_dynamic_texture(660, 257, dirtytexture, tag="dirty_texture")
    with dpg.texture_registry():
        dpg.add_dynamic_texture(660, 257, cleantexture, tag="clean_texture")
    dpg.create_viewport(title= 'Denoiser',width=690, height=800)
 

    with dpg.window(label= 'cleanup_a.0.6', width=678, height=760):
        dpg.add_image("dirty_texture")
        dpg.add_text("stft output")
        dpg.add_image("clean_texture")
        dpg.add_text("waveform out")
        dpg.add_simple_plot( min_scale=-1.0, max_scale=1.0, width=660, height=100, tag="plot2") 
        dpg.add_button(label="Disable", tag="toggleswitch", callback=denoisetoggle)
        dpg.add_button(label="Disable", tag="toggleswitch2", callback=entropytoggle)
    
    dpg.setup_dearpygui()
    dpg.configure_app(auto_device=True)

    dpg.show_viewport()
    dpg.start_dearpygui()
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

        
    close()  # clean up the program runtime when the user closes the window
   
