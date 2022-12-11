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

#How to use this file:
#step one: using https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Windows-x86_64.exe
#install python.
#step three: locate the dedicated python terminal in your start menu, called mambaforge prompt.
#within that prompt, give the following instructions:
#conda install pip numpy scipy
#pip install librosa tk dearpygui np-rw-buffer numba
#if all of these steps successfully complete, you're ready to go, otherwise fix things.
#run it with python demo_fileIO.py

from scipy.io import wavfile
from tkinter import Tk
from tkinter.filedialog import askopenfilename


import os
import numpy
import numba

from librosa import stft, istft
from time import time as time




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
        if entropy_unmasked[each] >0.0550346970932031:
            test = (entropy_unmasked[each]  - 0.0550346970932031) / (0.20608218909223255  - 0.0550346970932031)
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


@numba.jit()
def threshhold(arr):
  return (atd(arr)+ numpy.nanmedian(arr[numpy.nonzero(arr)])) 

def denoise(data: numpy.ndarray):

    #24000/256 = 93.75 hz per frequency bin.
    #a 4000 hz window(the largest for SSB is roughly 43 bins.
    #https://en.wikipedia.org/wiki/Voice_frequency
    #however, practically speaking, voice frequency cuts off just above 3400hz.
    #*most* SSB channels are constrained far below this.
    #to catch most voice activity on shortwave, we use the first 32 bins, or 3000hz.
    #we automatically set all other bins to the residue value.
    #reconstruction or upsampling of this reduced bandwidth signal is a different problem we dont solve here.
 
    data= numpy.asarray(data,dtype=float) #correct byte order of array   
    lettuce_euler_macaroni = 0.06 #use the euler/macaroni constant for noise similarity- truncate to 6 points of precision and raise by 1
    #this provides a safe constraint.

    boxcar = numpy.asarray([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
    hann = numpy.asarray([0.00000000e+00,3.77965773e-05,1.51180595e-04,3.40134910e-04,6.04630957e-04,9.44628746e-04,1.36007687e-03,1.85091253e-03,2.41706151e-03,3.05843822e-03,3.77494569e-03,4.56647559e-03,5.43290826e-03,6.37411270e-03,7.38994662e-03,8.48025644e-03,9.64487731e-03,1.08836332e-02,1.21963367e-02,1.35827895e-02,1.50427819e-02,1.65760932e-02,1.81824916e-02,1.98617342e-02,2.16135671e-02,2.34377255e-02,2.53339336e-02,2.73019047e-02,2.93413412e-02,3.14519350e-02,3.36333667e-02,3.58853068e-02,3.82074146e-02,4.05993391e-02,4.30607187e-02,4.55911813e-02,4.81903443e-02,5.08578147e-02,5.35931893e-02,5.63960544e-02,5.92659864e-02,6.22025514e-02,6.52053053e-02,6.82737943e-02,7.14075543e-02,7.46061116e-02,7.78689827e-02,8.11956742e-02,8.45856832e-02,8.80384971e-02,9.15535940e-02,9.51304424e-02,9.87685015e-02,1.02467221e-01,1.06226043e-01,1.10044397e-01,1.13921708e-01,1.17857388e-01,1.21850843e-01,1.25901469e-01,1.30008654e-01,1.34171776e-01,1.38390206e-01,1.42663307e-01,1.46990432e-01,1.51370928e-01,1.55804131e-01,1.60289372e-01,1.64825973e-01,1.69413247e-01,1.74050502e-01,1.78737036e-01,1.83472140e-01,1.88255099e-01,1.93085190e-01,1.97961681e-01,2.02883837e-01,2.07850913e-01,2.12862158e-01,2.17916814e-01,2.23014117e-01,2.28153297e-01,2.33333576e-01,2.38554171e-01,2.43814294e-01,2.49113148e-01,2.54449933e-01,2.59823842e-01,2.65234062e-01,2.70679775e-01,2.76160159e-01,2.81674384e-01,2.87221617e-01,2.92801019e-01,2.98411747e-01,3.04052952e-01,3.09723782e-01,3.15423378e-01,3.21150881e-01,3.26905422e-01,3.32686134e-01,3.38492141e-01,3.44322565e-01,3.50176526e-01,3.56053138e-01,3.61951513e-01,3.67870760e-01,3.73809982e-01,3.79768282e-01,3.85744760e-01,3.91738511e-01,3.97748631e-01,4.03774209e-01,4.09814335e-01,4.15868096e-01,4.21934577e-01,4.28012860e-01,4.34102027e-01,4.40201156e-01,4.46309327e-01,4.52425614e-01,4.58549094e-01,4.64678841e-01,4.70813928e-01,4.76953428e-01,4.83096412e-01,4.89241951e-01,4.95389117e-01,5.01536980e-01,5.07684611e-01,5.13831080e-01,5.19975458e-01,5.26116815e-01,5.32254225e-01,5.38386758e-01,5.44513487e-01,5.50633486e-01,5.56745831e-01,5.62849596e-01,5.68943859e-01,5.75027699e-01,5.81100196e-01,5.87160431e-01,5.93207489e-01,5.99240456e-01,6.05258418e-01,6.11260467e-01,6.17245695e-01,6.23213197e-01,6.29162070e-01,6.35091417e-01,6.41000339e-01,6.46887944e-01,6.52753341e-01,6.58595644e-01,6.64413970e-01,6.70207439e-01,6.75975174e-01,6.81716305e-01,6.87429962e-01,6.93115283e-01,6.98771407e-01,7.04397480e-01,7.09992651e-01,7.15556073e-01,7.21086907e-01,7.26584315e-01,7.32047467e-01,7.37475536e-01,7.42867702e-01,7.48223150e-01,7.53541070e-01,7.58820659e-01,7.64061117e-01,7.69261652e-01,7.74421479e-01,7.79539817e-01,7.84615893e-01,7.89648938e-01,7.94638193e-01,7.99582902e-01,8.04482319e-01,8.09335702e-01,8.14142317e-01,8.18901439e-01,8.23612347e-01,8.28274329e-01,8.32886681e-01,8.37448705e-01,8.41959711e-01,8.46419017e-01,8.50825950e-01,8.55179843e-01,8.59480037e-01,8.63725883e-01,8.67916738e-01,8.72051970e-01,8.76130952e-01,8.80153069e-01,8.84117711e-01,8.88024281e-01,8.91872186e-01,8.95660845e-01,8.99389686e-01,9.03058145e-01,9.06665667e-01,9.10211707e-01,9.13695728e-01,9.17117204e-01,9.20475618e-01,9.23770461e-01,9.27001237e-01,9.30167455e-01,9.33268638e-01,9.36304317e-01,9.39274033e-01,9.42177336e-01,9.45013788e-01,9.47782960e-01,9.50484434e-01,9.53117800e-01,9.55682662e-01,9.58178630e-01,9.60605328e-01,9.62962389e-01,9.65249456e-01,9.67466184e-01,9.69612237e-01,9.71687291e-01,9.73691033e-01,9.75623159e-01,9.77483377e-01,9.79271407e-01,9.80986977e-01,9.82629829e-01,9.84199713e-01,9.85696393e-01,9.87119643e-01,9.88469246e-01,9.89745000e-01,9.90946711e-01,9.92074198e-01,9.93127290e-01,9.94105827e-01,9.95009663e-01,9.95838660e-01,9.96592693e-01,9.97271648e-01,9.97875422e-01,9.98403924e-01,9.98857075e-01,9.99234805e-01,9.99537058e-01,9.99763787e-01,9.99914959e-01,9.99990551e-01,9.99990551e-01,9.99914959e-01,9.99763787e-01,9.99537058e-01,9.99234805e-01,9.98857075e-01,9.98403924e-01,9.97875422e-01,9.97271648e-01,9.96592693e-01,9.95838660e-01,9.95009663e-01,9.94105827e-01,9.93127290e-01,9.92074198e-01,9.90946711e-01,9.89745000e-01,9.88469246e-01,9.87119643e-01,9.85696393e-01,9.84199713e-01,9.82629829e-01,9.80986977e-01,9.79271407e-01,9.77483377e-01,9.75623159e-01,9.73691033e-01,9.71687291e-01,9.69612237e-01,9.67466184e-01,9.65249456e-01,9.62962389e-01,9.60605328e-01,9.58178630e-01,9.55682662e-01,9.53117800e-01,9.50484434e-01,9.47782960e-01,9.45013788e-01,9.42177336e-01,9.39274033e-01,9.36304317e-01,9.33268638e-01,9.30167455e-01,9.27001237e-01,9.23770461e-01,9.20475618e-01,9.17117204e-01,9.13695728e-01,9.10211707e-01,9.06665667e-01,9.03058145e-01,8.99389686e-01,8.95660845e-01,8.91872186e-01,8.88024281e-01,8.84117711e-01,8.80153069e-01,8.76130952e-01,8.72051970e-01,8.67916738e-01,8.63725883e-01,8.59480037e-01,8.55179843e-01,8.50825950e-01,8.46419017e-01,8.41959711e-01,8.37448705e-01,8.32886681e-01,8.28274329e-01,8.23612347e-01,8.18901439e-01,8.14142317e-01,8.09335702e-01,8.04482319e-01,7.99582902e-01,7.94638193e-01,7.89648938e-01,7.84615893e-01,7.79539817e-01,7.74421479e-01,7.69261652e-01,7.64061117e-01,7.58820659e-01,7.53541070e-01,7.48223150e-01,7.42867702e-01,7.37475536e-01,7.32047467e-01,7.26584315e-01,7.21086907e-01,7.15556073e-01,7.09992651e-01,7.04397480e-01,6.98771407e-01,6.93115283e-01,6.87429962e-01,6.81716305e-01,6.75975174e-01,6.70207439e-01,6.64413970e-01,6.58595644e-01,6.52753341e-01,6.46887944e-01,6.41000339e-01,6.35091417e-01,6.29162070e-01,6.23213197e-01,6.17245695e-01,6.11260467e-01,6.05258418e-01,5.99240456e-01,5.93207489e-01,5.87160431e-01,5.81100196e-01,5.75027699e-01,5.68943859e-01,5.62849596e-01,5.56745831e-01,5.50633486e-01,5.44513487e-01,5.38386758e-01,5.32254225e-01,5.26116815e-01,5.19975458e-01,5.13831080e-01,5.07684611e-01,5.01536980e-01,4.95389117e-01,4.89241951e-01,4.83096412e-01,4.76953428e-01,4.70813928e-01,4.64678841e-01,4.58549094e-01,4.52425614e-01,4.46309327e-01,4.40201156e-01,4.34102027e-01,4.28012860e-01,4.21934577e-01,4.15868096e-01,4.09814335e-01,4.03774209e-01,3.97748631e-01,3.91738511e-01,3.85744760e-01,3.79768282e-01,3.73809982e-01,3.67870760e-01,3.61951513e-01,3.56053138e-01,3.50176526e-01,3.44322565e-01,3.38492141e-01,3.32686134e-01,3.26905422e-01,3.21150881e-01,3.15423378e-01,3.09723782e-01,3.04052952e-01,2.98411747e-01,2.92801019e-01,2.87221617e-01,2.81674384e-01,2.76160159e-01,2.70679775e-01,2.65234062e-01,2.59823842e-01,2.54449933e-01,2.49113148e-01,2.43814294e-01,2.38554171e-01,2.33333576e-01,2.28153297e-01,2.23014117e-01,2.17916814e-01,2.12862158e-01,2.07850913e-01,2.02883837e-01,1.97961681e-01,1.93085190e-01,1.88255099e-01,1.83472140e-01,1.78737036e-01,1.74050502e-01,1.69413247e-01,1.64825973e-01,1.60289372e-01,1.55804131e-01,1.51370928e-01,1.46990432e-01,1.42663307e-01,1.38390206e-01,1.34171776e-01,1.30008654e-01,1.25901469e-01,1.21850843e-01,1.17857388e-01,1.13921708e-01,1.10044397e-01,1.06226043e-01,1.02467221e-01,9.87685015e-02,9.51304424e-02,9.15535940e-02,8.80384971e-02,8.45856832e-02,8.11956742e-02,7.78689827e-02,7.46061116e-02,7.14075543e-02,6.82737943e-02,6.52053053e-02,6.22025514e-02,5.92659864e-02,5.63960544e-02,5.35931893e-02,5.08578147e-02,4.81903443e-02,4.55911813e-02,4.30607187e-02,4.05993391e-02,3.82074146e-02,3.58853068e-02,3.36333667e-02,3.14519350e-02,2.93413412e-02,2.73019047e-02,2.53339336e-02,2.34377255e-02,2.16135671e-02,1.98617342e-02,1.81824916e-02,1.65760932e-02,1.50427819e-02,1.35827895e-02,1.21963367e-02,1.08836332e-02,9.64487731e-03,8.48025644e-03,7.38994662e-03,6.37411270e-03,5.43290826e-03,4.56647559e-03,3.77494569e-03,3.05843822e-03,2.41706151e-03,1.85091253e-03,1.36007687e-03,9.44628746e-04,6.04630957e-04,3.40134910e-04,1.51180595e-04,3.77965773e-05,0.00000000e+00])
    #predefining our windows also saves some cycles    

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
    stft_vh =(stft_vh -numpy.nanmin(stft_vh))/numpy.ptp(stft_vh) #normalize to 0,1

    stft_d = stft_vh[stft_vh<floor]
    stft_d = stft_d[stft_d>0]
    residue =  man(stft_d)

    stft_vr  = stft_vh[0:32,:].copy() #obtain the desired bins
    stft_vr = numpy.sort(stft_vr,axis=0) #sort the array
    ent_hann = fast_entropy(stft_vr)
    ent_hann = smoothpadded(ent_hann)

    minent = numpy.minimum(ent_hann,ent_box)
    maxent = numpy.maximum(ent_hann,ent_box)
    factor = numpy.max(maxent)

    if factor < lettuce_euler_macaroni: 
      stft_hann = stft_hann * residue
      processed = istft(stft_hann,window=hann)
      return processed

    entropy = (maxent+minent)/2
    entropy_unmasked = entropy.copy()
    entropy[entropy<lettuce_euler_macaroni] = 0
    entropy[entropy>0] = 1



    mask=numpy.zeros_like(stft_vh)
    thresh = threshhold(numpy.ravel(stft_vh[stft_vh>=floor])) - man(numpy.ravel(stft_vh[stft_vh>=floor]))

    mask[0:32,:] = fast_peaks(stft_vh[0:32,:],entropy,thresh,entropy_unmasked)
 
    mask = numpyfilter_wrapper_50(mask)
    mask=(mask-numpy.nanmin(mask))/numpy.ptp(mask)#correct basis    
    mask[mask==0] = residue 
    stft_hann = stft_hann * mask
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



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
# look for input file.wav
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename(filetypes=(('wav files', '*.wav'),), title='Select input file.wav')
    file_path, file_tail = os.path.split(filename)
    infile = filename
    outfile = f"{file_path}/cleanup, {file_tail}"
    print(infile)
    print(outfile)
    rate, data = wavfile.read(infile) # pcm16bit
    data_max = (2 ** 15 - 1)  # peak  in pcm16bit
  # peak level in pcm16bit
    data = data * 1.0 / data_max  # from pcm16bit to float (-1.0, 1.0)

    reduced_noise = process_data(data) *2.0  #  6 db gain
    # from float (-1.0, 1.0) to pcm16bit
    numpy.clip(reduced_noise, -1.0, 1.0)   # clip signal float range (1.0,-1-0)
    reduced_noise = reduced_noise * data_max
    wavfile.write(outfile, rate, reduced_noise.astype(numpy.int16))

