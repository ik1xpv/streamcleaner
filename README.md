# Stream Cleaner
Since i'm not good at explanations, I'll just keep this short and simple:

This repository contains a really good method for reducing noise in audio, for use with speech and other similar waveforms.
It can be used to denoise CW as well. It should not be used for denoising data modes.
It is meant for use on shortwave radio, with bandlimited single sideband signals with bandwidth of 4000hz or less.
It assumes the speech bandwidth is 3000hz or less.

It works on the basis of logit comparisons for time bins and thresholding using robust measures, with minimal dependencies.
This code is MIT and GPL licensed.


