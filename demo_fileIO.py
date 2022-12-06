'''
Demo v1.1
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


import os
import numpy

from time import sleep
from time import time as time
from librosa import stft, istft
from scipy.io import wavfile
from scipy.special import logit
from scipy import interpolate as interp
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def hann(M, sym=True):
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

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


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


def entropy(data: numpy.ndarray):
    a = data.copy()
    a = numpy.sort(a)
    scaled = numpy.interp(a, (a.min(), a.max()), (-6, +6))
    fprint = numpy.linspace(0, 1, a.size)
    y = logit(fprint)
    y[y == -numpy.inf] = -6
    y[y == +numpy.inf] = 6
    z = numpy.corrcoef(scaled, y)
    completeness = z[0, 1]
    sigma = 1 - completeness
    return sigma


def numpy_adjacent_filter(data: numpy.ndarray):
   normal = data.copy()
   transposed = data.copy()
   transposed = transposed.T
   transposed_raveled = numpy.ravel(transposed)
   normal_raveled = numpy.ravel(normal)
   zeroth = numpy.zeros_like(normal_raveled)

   A = numpy.zeros_like(transposed_raveled)
   B = numpy.ravel(transposed_raveled[1:-2])
   C = numpy.ravel(transposed_raveled[2:-1])
   A[1:-2] = transposed_raveled[1:-2] + B[:] + C[:]
   A[0] = (transposed_raveled[0] + transposed_raveled[1])/2
   A[-1] = (transposed_raveled[-1] + transposed_raveled[-2])/2
   transposed_raveled[:] = A/3
   transposed = transposed.T

   A = numpy.zeros_like(normal_raveled)
   B = numpy.ravel(normal_raveled[1:-2])
   C = numpy.ravel(normal_raveled[2:-1])
   A[1:-2] = normal_raveled[1:-2] + B[:] + C[:]
   A[0] = (normal_raveled[0] + normal_raveled[1])/2
   A[-1] = (normal_raveled[-1] + normal_raveled[-2])/2
   normal_raveled[:] = A/3

   return (transposed + normal) /2

def numpyfilter_wrapper_50(data: numpy.ndarray):
  d = data.copy()
  for i in range(50):
    d = numba_adjacent_filter(d)
  return d

def denoise(data: numpy.ndarray):
    data= numpy.asarray(data,dtype=float) #correct byte order of array   

    stft_r = stft(data,n_fft=512,window=boxcar) #get complex representation
    stft_vr = numpy.square(stft_r.real) + numpy.square(stft_r.imag) #obtain absolute**2
    Y = stft_vr[~numpy.isnan(stft_vr)]
    max = numpy.where(numpy.isinf(Y),0,Y).argmax()
    max = Y[max]
    stft_vr = numpy.nan_to_num(stft_vr, copy=True, nan=0.0, posinf=max, neginf=0.0)#correct irrationalities
    stft_vr = numpy.sqrt(stft_vr) #stft_vr >= 0 
    stft_vr=(stft_vr-numpy.nanmin(stft_vr))/numpy.ptp(stft_vr)
    ent = numpy.apply_along_axis(func1d=entropy,axis=0,arr=stft_vr[0:32,:]) #32 is pretty much the speech cutoff?
    #adjust this further to adapt to the width of your filter.
    trend = moving_average(ent,20)
    factor = numpy.max(trend)
    ent=(ent-numpy.nanmin(ent))/numpy.ptp(ent)#correct basis 
    t1 = atd(ent)
    ent[ent<t1] = 0
    ent[ent>0] = 1

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


    t = threshhold(stft_vr[stft_vr>=t])   #obtain the halfway threshold
    mask_two = numpy.where(stft_vr>=t/2, 1.0,0)

    arr1_interp =  interp.interp1d(numpy.arange(ent.size),ent)
    ent = arr1_interp(numpy.linspace(0,ent.size-1,stft_r.shape[1]))



    mask = mask_two * ent[None,:] #remove regions from the mask that are noise
    mask[mask==0] = r #reduce warbling, you could also try r/2 or r/10 or something like that, its not as important
    mask = numpyfilter_wrapper_50(mask)
    if factor < 0.0777: #unknown the exact most precise, correct option
      mask[:] = r #there is no signal here, and therefore, there is no point in attempting to mask.
    #we now have two filters, and we should select criteria among them
    mask=(mask-numpy.nanmin(mask))/numpy.ptp(mask)#correct basis    

     
    stft_r = stft_r * mask
 
    processed = istft(stft_r,window=hann)
    return processed


def padarray(A, size):
    t = size - len(A)
    return numpy.pad(A, pad_width=(0, t), mode='constant',constant_values=numpy.std(A))

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def process_data(data: numpy.ndarray):
    print("processing ", data.size / rate, " seconds long file at ", rate, " rate.")
    start = time()
    processed = []
    for each in chunks(data, rate):
        if each.size == rate:
            processed.append(denoise(each))
        else:
            psize = each.size
            working = padarray(each, rate)
            working = denoise(working)
            processed.append(working[0:psize])
    end = time()
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

