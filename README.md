# Stream Cleaner
Since i'm not good at explanations, I'll just keep this short and simple:

Cleanup.py is the file you want here.

This repository contains a really good method for reducing noise in audio, for use with speech and other similar waveforms.
It can be used to denoise CW as well, although it should be tweaked for that. It should not be used for denoising data modes.
It is meant for use on shortwave radio, with bandlimited single sideband signals with bandwidth of 4000hz or less.
It assumes the speech bandwidth is 3000hz or less. It does not work on a theoretical basis strongly supported, although 
contributions from an engineer to derive such a theoretical basis for this work is strongly interesting to us as this was devised solely empirically.
More explanations in the algorithm explanation file, however: the basic gist of it is this:

It works on the basis of logit comparisons for time bins and thresholding using robust measures, with minimal dependencies.
This code is MIT and GPL licensed, depending on numba, numpy, some library's stfts, and, for specific use cases,
invoking dearpygui, tk, pyaudio, and other components as needed, but they are not required for the core algorithm to function.
It would be easy to port this algorithm to other languages provided you have a numpy-equivalent array handling library and stft.
OLA-consistent processing is provided by pyroomacoustics, but more scientifically precise analysis windows are provided by ssqueezepy.

Most of the programs provided here require a minimum of a 1 second frame to deliver good results.
That means they are not quite realtime but 1 second delayed or process in a 1 second frame.
You can also try realtime.py for a version that considers 180ms fragments but I do not consider it great.


