# Stream Cleaner
Since i'm not good at explanations, I'll just keep this short and simple:

This repository contains a really good method for reducing noise in audio, for use with speech and other similar waveforms.
It can be used to denoise CW as well, although it should be tweaked for that. It should not be used for denoising data modes.
It is meant for use on shortwave radio, with bandlimited single sideband signals with bandwidth of 4000hz or less.
It assumes the speech bandwidth is 3000hz or less. It does not presently include a means for eliminating discongruities at the edges,
between consecutive seconds of data, although the correct solution is greatly of interest to us.

It works on the basis of logit comparisons for time bins and thresholding using robust measures, with minimal dependencies.
This code is MIT and GPL licensed, depending on numba, numpy, librosa's stft, and, for specific use cases,
invoking dearpygui, tk, pyaudio, and other components as needed, but they are not required for the core algorithm to function.
It would be easy to port this algorithm to other languages provided you have a numpy-equivalent array handling library and stft.

