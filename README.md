# Stream Cleaner
Since i'm not good at explanations, I'll just keep this short and simple:

This repository contains a really good method for reducing noise in audio, for use with speech and other similar waveforms.
It can be used to denoise CW as well. It should not be used for denoising data modes.

It works on the basis of logit comparisons for time bins and thresholding using robust measures.
It relies on librosa's fast stft and numpy- and pyaudio and dearpygui on windows, but they are not necessary for the algorithm itself, only for the "real time delayed" program that will process your audio in realtime.
The demonstration file processing program also relies on tkinter and scipy for a simple interface.
In terms of statistical measures, I couldn't tell you what the STOI or the MSE or PSEQ are.
But I can say this- it gives really good results, regardless of how noisy your audio is.
These results may very well be as good or better than commercial, state of the art denoising software.
In fact, they might as well close down their businesses entirely, considering this is MIT and GPL licensed.
Other contributors provided assistance with developing components for the real time demo and the file processing demo, their work is theirs to share or withhold.

As for the core denoising algorithm, which I obsessively, neurotically fettered over and iterated over through a great deal of trial, pain and effort, unemployed, on food stamps, with only a marginal understanding of programming, and certainly no strong formal background:

If you use this commercially in a for-profit setting, I expect but do not legally obligate you to compensate me.
I legally exempt anyone using the MIT and GPL license from any liability for the use, modification, or commercialization of this denoising approach, but I will certainly testify before god that you have stolen my time and effort from me.

The algorithm itself seems simple enough, and that's how most things in life are- you try many different things
until you find what works, and often enough as of not it's just a combination you hadn't considered before.
All I really wanted from it was a way to squelch intelligently and better extract voice.
None of this incorporated things I learned directly from other's research papers, but all of them influenced me and
if you look in the history you will see i attempted to implement a bunch of other methods to see what they could do.
This work is a testament to the value of effort and determination moderated by patience and dilligence.
If I can do it, you can do it.


