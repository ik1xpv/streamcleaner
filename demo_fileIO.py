#This program will crash if you feed it stereo. Feel free to submit a patch to add stereo handling.
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
#pip install tk numba pyroomacoustics ssqueezepy 
#if all of these steps successfully complete, you're ready to go, otherwise fix things.
#run it by executing this command in the terminal, in the folder containing the file: python demo_fileIO.py

from scipy.io import wavfile
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import time

import os
import numba
import numpy
from ssqueezepy import stft as sstft
import pyroomacoustics as pra
from threading import Thread

import pyaudio
import dearpygui.dearpygui as dpg
os.environ['SSQ_PARALLEL'] = '0'

@numba.jit()
def man(arr):
    med = numpy.nanmedian(arr[numpy.nonzero(arr)])
    return numpy.nanmedian(numpy.abs(arr - med))

@numba.jit()
def atd(arr):
    x = numpy.square(numpy.abs(arr - man(arr)))
    return numpy.sqrt(numpy.nanmean(x))

@numba.jit()
def threshold(data: numpy.ndarray):
 return numpy.sqrt(numpy.nanmean(numpy.square(numpy.abs(data -numpy.nanmedian(numpy.abs(data - numpy.nanmedian(data[numpy.nonzero(data)]))))))) + numpy.nanmedian(data[numpy.nonzero(data)])

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

def moving_average(x, w):
    return numpy.convolve(x, numpy.ones(w), 'same') / w

def smoothpadded(data: numpy.ndarray):
  o = numpy.pad(data, data.size//2, mode='median')
  return moving_average(o,14)[data.size//2: -data.size//2]

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


@numba.njit()
def fast_entropy(data: numpy.ndarray):
   logit = numpy.asarray([0.,0.08805782,0.17611565,0.22947444,0.2687223,0.30031973,0.32715222,0.35076669,0.37209427,0.39174363,0.41013892,0.42759189,0.44434291,0.46058588,0.47648444,0.4921833,0.5078167,0.52351556,0.53941412,0.55565709,0.57240811,0.58986108,0.60825637,0.62790573,0.64923331,0.67284778,0.69968027,0.7312777,0.77052556,0.82388435,0.91194218,1.])
   #note: if you alter the number of bins, you need to regenerate this array.
   entropy = numpy.zeros(data.shape[1])
   for each in numba.prange(data.shape[1]):
      d = data[:,each]
      d = numpy.interp(d, (d[0], d[-1]), (0, +1))
      entropy[each] = 1 - numpy.corrcoef(d, logit)[0,1]
   return entropy


@numba.jit()
def fast_peaks(stft_:numpy.ndarray,entropy:numpy.ndarray,thresh:numpy.float64,entropy_unmasked:numpy.ndarray):
    #0.01811 practical lowest - but the absolute lowest is 0. 0 is a perfect logistic. 
    #0.595844362 practical highest - an array of zeros with 1 value of 1.
    #note: this calculation is dependent on the size of the logistic array.
    #if you change the number of bins, this constraint will no longer hold and renormalizing will fail.
    #a lookup table is not provided, however the value can be derived by setting up a one-shot fast_entropy that only considers one array,
    #sorting, interpolating, and comparing it. 
    mask = numpy.zeros_like(stft_)
    for each in numba.prange(stft_.shape[1]):
        data = stft_[:,each]
        if entropy[each] == 0:
            mask[0:32,each] =  0
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
        mask[0:32,each] = data[:]
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
    #*most* SSB channels are constrained far below this.
    #to catch most voice activity on shortwave, we use the first 32 bins, or 3000hz.
    #we automatically set all other bins to the residue value.
    #reconstruction or upsampling of this reduced bandwidth signal is a different problem we dont solve here.
 
    data= numpy.asarray(data,dtype=float) #correct byte order of array if it is incorrect
    lettuce_euler_macaroni = 0.057

    stft_logit = sstft(x=data,window=logit_window,n_fft=512,hop_len=128)
    stft_vl =  numpy.abs(stft_logit) #returns the same as other methods
    stft_hann = sstft(x=data,window=hann,n_fft=512,hop_len=128) #get complex representation
    stft_vh =  numpy.abs(stft_hann) #returns the same as other methods

    stft_vl  = stft_vl[0:32,:] #obtain the desired bins
    stft_vl = numpy.sort(stft_vl,axis=0) #sort the array
    
    entropy_unmasked = fast_entropy(stft_vl)
    entropy = smoothpadded(entropy_unmasked)
    factor = numpy.max(entropy)

    if factor < lettuce_euler_macaroni: 
      return stft_vh.T * 1e-6
    

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
    # an ionosound sweep is also around or better than 24 samples, also
    # number of bins and streak doubled for hop size 64
    # for hop size = 128, nbins = 22, maxstreak = 16
    #if hop size 256 was chosen, nbins would be 11 and maxstreak would be 11.
    #however, fs/hop = maximum alias-free frequency. For hop size of 64, it's over 300hz.
    if nbins<22 and maxstreak<16:
      return stft_vh.T * 1e-6
          
    mask=numpy.zeros_like(stft_vh)
    thresh = threshold(stft_vh[stft_vh>man(stft_vl.flatten())])
    mask[0:32,:] = fast_peaks(stft_vh[0:32,:],entropy,thresh,entropy_unmasked)
    mask[mask<1e-6] = 1e-6 #fill the residual with a small value

    
    mask1 = numpyfilter_wrapper_50(mask)
    mask = numpy.maximum(mask,mask1)#preserve peaks, flood-fill valley
    return mask.T


class FilterThread(Thread):
    def __init__(self):
        super(FilterThread, self).__init__()
        self.running = True
        self.NFFT = 512 #note: if you change window size or hop, you need to re-create the logistic window and hann windows used.
        self.NBINS=32
        self.hop = 128
        self.hann = hann
        self.synthesis = pra.transform.stft.compute_synthesis_window(self.hann, self.hop)
        self.stft = pra.transform.STFT(N=512, hop=self.hop, analysis_window=self.hann,synthesis_window=self.synthesis ,online=True)
        self.residual = 0

    def process(self,data):        
           self.stft.analysis(data)
           mask = mask_generate(data)
           output = self.stft.synthesis(self.stft.X* mask)
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
    reduced_noise = process_data(data)

    wavfile.write(outfile, rate, reduced_noise.astype(numpy.float32))
