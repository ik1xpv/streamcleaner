#a collection of the functions perhaps to be used and a rough draft of how this will work.
    
boxcar = numpy.ones(fft_len) * 0.5
hann = pra.hann(fft_len, flag='asymmetric', length='full')

def runningMeanFast(x, N):
    return numpy.convolve(x, numpy.ones((N,))/N,mode="valid")

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

def atomic_entropy(logit: numpy.ndarray, hann: numpy.ndarray, boxcar: numpy.ndarray):
#this function should be fed three arrays of an identical size.
#the logistic function should be generated ahead of time to minimize compute.
#this function should not be numba optimized for the interactive implementation, as it must be compiled with nuitka.
   #logit = generate_reasonable_logistic(NBINS) #produces a normalized logit
   hann = numpy.sort(hann)
   hann = numpy.interp(hann, (hann[0], hann[-1]), (0, +1))
   boxcar = numpy.sort(boxcar)
   hann = numpy.interp(boxcar, (boxcar[0], boxcar[-1]), (0, +1))
   return ((1 - numpy.corrcoef(hann, logit)[0,1]) + (1 - numpy.corrcoef(boxcar, logit)[0,1]))/2
   #returns the pre-smoothed, averaged individual time bin entropy calculation

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

def entropy_gate_and_smoothing(entropy: numpy.ndarray, entropy_constant: float)
    #9 *4 = 27ms before, after, and during the filter.
    smoothed = numpy.convolve(entropy, numpy.ones(9), 'same') / 9
    smoothed_gated = smoothed.copy()
    smoothed_gated[smoothed_gated< entropy_constant] = 0
    smoothed_gated[smoothed_gated>0] = 1
    total = numpy.sum(smoothed_gated)
    streak = longest_consecutive(smoothed_gated)
    #required total = 14
    #required streak = 11
    if total < 14 and streak <11: #there is probabilistically no signals present here.
        return numpy.zeros(9)
    else:
        return smoothed_gated[9:18]

def atomic_mask(hann: numpy.ndarray): 
    #this function should be fed [0:nbins] of one time bin, inserted to an array of zeros, appended to the mask.
    threshold = numpy.sqrt(numpy.nanmean(numpy.square(numpy.abs(hann - man(hann)))))
    hann[hann<threshold] = 0
    hann[hann>0] = 1
    return hann
     
def smooth_mask(mask: numpy.ndarray,entropy:numpy.ndarray: residue_constant: float):
    #this function should be fed stft_heightx27 samples as an array
    #and 27 samples as a entropy array
    #it should also be fed the residual constant, at this time determined manually.
    smoothed = smoothed * entropy[None,:] #i think this is right. you need to add another
    #dimension to the entropy array so it aligns with the mask
    smoothed = numpyfilter_wrapper_50(mask)
    smoothed = (smoothed - numpy.nanmin(smoothed))/numpy.ptp(smoothed)
    smoothed = smoothed[:,9:18]
    smoothed[smoothed==0] = residue_constant
    return smoothed


#Tenative working process:
#first, 
                              frames_per_buffer=int(self._processing_size),
                              stream_callback=self.non_blocking_stream_read

#processing size is set to 27.
#27 samples at a time become 9 bins.

#secondly:
 def non_blocking_stream_read(self, in_data, frame_count, time_info, status):
        audio_in = numpy.ndarray(buffer=in_data, dtype=self.dtype,
                                            shape=[int(self._processing_size * self._channels)]).reshape(-1,
                                                                                                         self.channels)
        audio_in_left = audio_in[:,0]
        audio_in_right = audio_in[:,0]
       
        #boxcar and hann analysis is done here.
        #one OLA stft, one mask, and one entropy array are written to outside of this function- for each channel.
        #these can be written as two 2d ringbuffers.

        #for each time bin, atomic_entropy and atomic mask is called.
        #these two sets of values should be put together into a buffer.



        return None, pyaudio.paContinue

#thirdly:

    def non_blocking_stream_write(self, in_data, frame_count, time_info, status):
       chans = []
        #for each channel:
            bins = self.stft_rb.read(9)
            #if the current entropy array size is less than 27, chans.append(stft.synthesis(bins).
              #otherwise:  mask and entropy = self.rb.get_data(27)
              # call entropy smoothing calculation on the entropy slice from the array.
              #if it's all zeros, chans.append(stft.synthesis(bins*residue).
              #otherwise: call mask smoothing function, on the first 27 values in the mask array.
              chans.append(stft.synthesis(bins*mask)) #for each channel. ie stft_left, stft_right
              #finally, advance the index: self.rb.read(9).
                
        return numpy.column_stack(chans).astype(self.dtype).tobytes(), pyaudio.paContinue
   
