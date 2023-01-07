#A Really crappy scientific article
Instructions to assist implementation in other languages and to facilitate the comprehension of the denoising method depicted here.

Caveat: This algorithm evolved through a process of trial and error. That is to say, it was achieved by empirical process and not built from theory.
As a result, everything offered here is without warrantee of suitability for any purpose, and no strong theoretical basis will be offered.
If any qualified engineer would like to contribute a strong theoretical support and help publish this research, their contribution and 
participation in the project would be warmly welcomed.

The denoising method depicted here consists of a masking process that considers two dimensional aspects of the information:
the amplitude and the noise similarity of a single column of the STFT representation, here called a time-bin.
A discrete STFT window consists of rows which contain amplitude of a given frequency sampled at points in time.

##Entropy
Noise fluctuates continually and in the shortwave spectrum typically has a guassian similarity, including impulse noise such as
that from man-made sources and incident weather. Even when it is not completely-guassian like, the similarity to guassian noise can be calculated. 

To do so, a specific, 1d frame, or 2d patch, of noise samples can be considered. Because noise fluctuates much more in time than in frequency,
it is recommended to sample a 1d frame in time. The variance of the samples in the frame, when truly noise, will not contain any significant features.
For this reason it is recommended to sample a frame only as wide as the channel you wish to consider. In our implementation, which consists of sampling
just over 3000hz for a single voice channel, using a FFT of 512 bins, the height of the time-bin is from 0hz to 3000hz and 32 bins starting at 0.

This produces for any individual moment in time a single array of 32 floating point numbers. 
For our calculations, we also use a custom window derived from a mirrored logit distribution which will be described in a moment.
It is assumed you are familiar with how to find the magnitude representation of a complex value, in python conveniently implemented as numpy.abs().

This array of 32 numbers is thusly sorted and then interpolated to the same range as the logit distribution, nominally 0 and 1.
It is then compared to said distribution and the pearson coefficient taken. In some implementations "N" is required- N is 32.
In python, an array of coefficient values is returned, a pair of "1" and some fraction, we take the fraction.

The larger this fraction is, the more similar the array is to a logit distribution or to a pure guassian noise distribution.
We subsequently subtract it from 1 to come up with a very small number. The smaller this number is, the more likely it is noise.
We define the noise similarity constraint emperically as the gompertz constant divided by 10, this number may be different for 
different array sizes, but it works suitably well for an array of 32.

#Thresholding
In our code, we compute several different thresholds that we call ATD, MAN, and THRESHHOLD.
I will not be elucidating on them here, as they are perfectly sensible to comprehend and completely without any merit.
I also derive a single statistical measure from the STFT representation used in entropy computation, called residue.

##combination
After padding, or finding some other way to reduce edge discongruities, and then smoothing an array of entropy values calculated over all time bins,
some rudimentary effort to determine if voice activity is likely present can be conducted counting the number of bins meeting the noise threshold constant
and the largest continual stretch of time meeting the constant. This can be done to reduce further computation.
Another routine then combines all the information determined thus far and amplitude-masks individual time-bins.

##Complications
The attempts depicted in this repository utilize either a 1 second or a somewhat large, around 150ms, window of samples. The larger the window, the better.
Smaller windows were attempted without good results. Other limitations exist and are commented in the code. However, the crux of what makes this approach
original and with merit is explained here, and you will find this denoising algorithm mostly immune to impulse noise and providing a most-excellent squelch.

##The real logit distribution
Since the logistic distribution endpoints are infinity, it is necessary to arbitrarily terminate them.
The logistic distribution utilized here is generated as follows:
First, an evenly spaced set of values from 0 to 1 is generated, the size of the distribution you want.
Here, it is 32.
Secondly, all but the first and the last are divided by 1 minus all but the first and last.
That is to say, it is an elementwise operation.
Thirdly, the last is set to twice the next to last minus the third from last.
Finally, the first is set to the negative of the last.

For the logistic window, 1 window equal to half the size, rounded down,
and if even, mirrored and combined, and if odd, the last value removed, then mirrored.
The behavior is virtually the same, if not fully identical.

You will note this window is quite smooth and nice and in some respects the opposite of the hann.

