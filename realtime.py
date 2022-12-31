print("Attention: this file does not work correctly. I am still debugging it. Please do not run this file unless prompted.")
exit(0)



#debugging still in progress
"""
Copyright 2022 Joshuah Rainstar, Oscar Steila

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
"""
Copyright 2022 Joshuah Rainstar, Oscar Steila 

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
"""

# cleanup_realtime1.1.8.py 12/30/2022

import numpy
import pyroomacoustics as pra
import pyaudio

from np_rw_buffer import RingBuffer
import dearpygui.dearpygui as dpg
from matplotlib import pyplot as plt

def runningMeanFast(x, N):
    return numpy.convolve(x, numpy.ones((N,)) / N, mode="valid")


def numpy_convolve_filter(data: numpy.ndarray):
    normal = data.copy()
    transposed = data.copy()
    transposed = transposed.T
    transposed_raveled = numpy.ravel(transposed)
    normal_raveled = numpy.ravel(normal)

    A = runningMeanFast(transposed_raveled, 3)
    transposed_raveled[0] = (transposed_raveled[0] + (transposed_raveled[1] + transposed_raveled[2]) / 2) / 3
    transposed_raveled[-1] = (transposed_raveled[-1] + (transposed_raveled[-2] + transposed_raveled[-3]) / 2) / 3
    transposed_raveled[1:-1] = A
    transposed = transposed.T

    A = runningMeanFast(normal_raveled, 3)
    normal_raveled[0] = (normal_raveled[0] + (normal_raveled[1] + normal_raveled[2]) / 2) / 3
    normal_raveled[-1] = (normal_raveled[-1] + (normal_raveled[-2] + normal_raveled[-3]) / 2) / 3
    normal_raveled[1:-1] = A
    return (transposed + normal) / 2


def numpyfilter_wrapper_50(data: numpy.ndarray):
    d = data.copy()
    for i in range(50):
        d = numpy_convolve_filter(d)
    return d


def atomic_entropy(logit: numpy.ndarray, hann: numpy.ndarray, boxcar: numpy.ndarray):
    # this function should be fed three arrays of an identical size. the logistic function should be generated ahead
    # of time to minimize compute. this function should not be numba optimized for the interactive implementation,
    # as it must be compiled with nuitka. logit = generate_reasonable_logistic(NBINS) #produces a normalized logit
    hann = numpy.sort(hann)
    hann = numpy.interp(hann, (hann[0], hann[-1]), (0, +1))
    boxcar = numpy.sort(boxcar)
    boxcar = numpy.interp(boxcar, (boxcar[0], boxcar[-1]), (0, +1))
    return ((1 - numpy.corrcoef(hann, logit)[0, 1]) + (1 - numpy.corrcoef(boxcar, logit)[0, 1])) / 2
    # returns the pre-smoothed, averaged individual time bin entropy calculation


def longestConsecutive(nums):
    streak = 0
    prevstreak = 0
    for num in range(nums.size):
        if nums[num] == 1:
            streak += 1
        if nums[num] == 0:
            prevstreak = max(streak, prevstreak)
            streak = 0
    return max(streak, prevstreak)


def entropy_gate_and_smoothing(entropy: numpy.ndarray, entropy_constant: float):
    # 9 *4 = 27ms before, after, and during the filter.
    smoothed = numpy.convolve(entropy, numpy.ones(9), 'same') / 9
    smoothed_gated = smoothed.copy()
    smoothed_gated[smoothed_gated < entropy_constant] = 0
    smoothed_gated[smoothed_gated > 0] = 1
    total = numpy.sum(smoothed_gated)
    streak = longestConsecutive(smoothed_gated)
    # required total = 14
    # required streak = 11
    if total < 14 and streak < 11:  # there is probabilistically no signals present here.
        return numpy.zeros(entropy.size)
    else:
        return smoothed_gated


def atomic_mask(hann: numpy.ndarray):
    # this function should be fed [0:nbins] of one time bin, inserted to an array of zeros, appended to the mask.
    threshold = numpy.sqrt(numpy.nanmean(
        numpy.square(numpy.abs(hann - numpy.nanmedian(numpy.abs(hann - numpy.nanmedian(hann[numpy.nonzero(hann)])))))))
    hann[hann < threshold] = 0
    hann[hann > 0] = 1
    return hann


def iterated_mask_generation(logit: numpy.ndarray, hann: numpy.ndarray, boxcar: numpy.ndarray, NBINS):
    mask = numpy.zeros((hann.shape[0], hann.shape[1] + 1))
    for each in range(hann.shape[0]):
        mask[each, :NBINS] = atomic_mask(hann[each, :NBINS])
        mask[each, -1] = atomic_entropy(logit, hann[each, :NBINS], boxcar[each, :NBINS])

    return mask  # this should return a 9x130 array with the entropy in the final row


def smooth_mask(mask: numpy.ndarray, residue_constant: float, entropy_constant: float):
    entropy = mask[:, -1]  # should be 27 samples
    entropy = entropy_gate_and_smoothing(entropy, entropy_constant)
    mask = mask[:, :-1]
    smoothed = mask * entropy[:,None]

    smoothed = numpyfilter_wrapper_50(smoothed)
    smoothed = (smoothed - numpy.nanmin(smoothed)) / numpy.ptp(smoothed)
    smoothed = smoothed[9:18, :]
    smoothed[smoothed == 0] = residue_constant
    return smoothed


def generate_reasonable_logistic(points):
    fprint = numpy.linspace(0.0, 1.0, points)
    fprint[1:-1] /= 1 - fprint[1:-1]
    fprint[1:-1] = numpy.log(fprint[1:-1])
    endpoint = 0.0
    counter = 0
    lookup = [2.1971892581166510, 4.5955363239333611, 6.9050175677146246, 9.2135681041095818, 11.5105362150512871,
              13.8139732623858311, 16.1212093471596063, 18.4212471940583633, 20.7199292066937240, 23.0260708890764363,
              25.3318791223971402, 27.6293850468557025, 29.9308315624775787, 32.2405437938828072, 34.5359179427814524,
              36.8424085116093920, 39.1437375251989579, 41.4465512066672090, 53.5009414419764653, 166.1841396146919578,
              874.5562803894281387, 4211.3929192572832108, 17116.3403342720121145, 60318.5852559730410576,
              189696.3978051021695137, 543994.4244986772537231, 1445156.3948460221290588, 3598799.5093398690223694,
              8477663.0022574663162231, 19027737.8391044139862061, 40926016.8621466159820557, 84755565.9826140403747559,
              169668882.8905963897705078, 329412845.1023845672607422, 622026323.3318710327148438,
              1145145647.3590965270996094, 2059730309.2295989990234375, 3626238193.6844787597656250,
              6258947560.4355010986328125, 10606383754.8501892089843750, 17668841282.3623657226562500,
              28968020901.3501586914062500, 46789106044.4216308593750000, 74522528302.5477294921875000,
              117141641713.1768798828125000, 181864057501.9619140625000000, 279059112640.5244140625000000,
              423482615174.2539062500000000, 635943537379.3251953125000000, 945536803119.7412109375000000]
    endpoint = lookup[0]
    while endpoint < fprint[-2]:
        counter = counter + 1
        endpoint = lookup[counter]
        if endpoint >= fprint[-2]:
            break

    fprint[-1] = lookup[counter + 1]
    fprint[0] = -fprint[-1]
    fprint = numpy.interp(fprint, (fprint[0], fprint[-1]), (0, 1))
    return fprint


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

    def __init__(self, sample_rate=16000, channels=2,  micindex=1, speakerindex=1, dtype=numpy.float32):
        self.pa = pyaudio.PyAudio()
        self.fft_len = 256
        self.hop_ratio = 4
        self.NBINS = 46  # todo: add a helper routine to set this based on a frequency selected by the user for a lowpass window
        self.hop_size = self.fft_len // self.hop_ratio
        self.hann_window = pra.hann(self.fft_len, flag="asymetric", length="full")
        self.boxcar_window = numpy.ones(self.fft_len) * 0.5
        self.synthesis_window = pra.transform.stft.compute_synthesis_window(self.hann_window, self.hop_size)
        self._processing_size = ((self.fft_len * 2) + self.hop_size)
        self.stft_hann = pra.transform.STFT(self.fft_len, hop=self.hop_size, analysis_window=self.hann_window,
                                            synthesis_window=self.synthesis_window, streaming=True, num_frames=9)
        self.stft_boxcar = pra.transform.STFT(self.fft_len, hop=self.hop_size, analysis_window=self.boxcar_window,
                                              streaming=True, num_frames=9)
        self.logit = generate_reasonable_logistic(self.NBINS)
        self._sample_rate = sample_rate
        self._channels = channels
        self.channels = channels
        self.residue = 0.01
        self.entropy_constant = 0.0074
        self.mask_buffer_left = RingBuffer(36, 130)

        self.stft_buffer_left = RingBuffer(36, 129)  # we need 18 x 129
        #this will write in chunks of 9.
        self.stft_buffer_left.dtype = numpy.complex128
        self.mask_buffer_old_left = numpy.ones((9,130),dtype=numpy.float64)

        self.mask_buffer_right = RingBuffer(36, 130)
        self.stft_buffer_right = RingBuffer(36, 129)
        self.stft_buffer_right.dtype = numpy.complex128
        self.mask_buffer_old_right = numpy.ones((9, 130), dtype=numpy.float64)
        self.micindex = micindex
        self.speakerindex = speakerindex
        self.micstream = None
        self.speakerstream = None
        self.speakerdevice = ""
        self.micdevice = ""
        self.dtype = dtype

        # Set inputs for inheritance

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

        stream = self.pa.open(format=pyaudio.paFloat32,
                              channels=self._channels,
                              rate=int(self._sample_rate),
                              input=True,
                              input_device_index=self.micindex,  # device_index,
                              # each frame carries twice the data of the frames
                              frames_per_buffer=int(self._processing_size),
                              stream_callback=self.non_blocking_stream_read,
                              start=True  # Need start to be False if you don't want this to start right away
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

        stream = self.pa.open(format=pyaudio.paFloat32,
                              channels=self._channels,
                              rate=int(self._sample_rate),
                              output=True,
                              output_device_index=self.speakerindex,
                              frames_per_buffer=int(self._processing_size),
                              stream_callback=self.non_blocking_stream_write,
                              start=True  # Need start to be False if you don't want this to start right away
                              )
        return stream

    # it is critical that this function do as little as possible, as fast as possible. numpy.ndarray is the fastest we can move.
    # attention: numpy.ndarray is actually faster than frombuffer for known buffer sizes
    def non_blocking_stream_read(self, in_data, frame_count, time_info, status):
        audio_in = numpy.ndarray(buffer=in_data, dtype=self.dtype,
                                 shape=[int(self._processing_size * self._channels)]).reshape(-1,
                                                                                              self.channels)

        audio_in_left = audio_in[:, 0]
        boxcar = self.stft_boxcar.analysis(audio_in_left)
        hann = self.stft_hann.analysis(audio_in_left)
        mask = iterated_mask_generation(self.logit, numpy.abs(hann), numpy.abs(boxcar), self.NBINS)
        self.mask_buffer_left.expanding_write(mask,error=False)
        self.stft_buffer_left.expanding_write(hann,error=False)

        audio_in_right = audio_in[:, 1]
        boxcar = self.stft_boxcar.analysis(audio_in_right)
        hann = self.stft_hann.analysis(audio_in_right)
        mask = iterated_mask_generation(self.logit, numpy.abs(hann), numpy.abs(boxcar), self.NBINS)
        self.mask_buffer_right.expanding_write(mask,error=False)
        self.stft_buffer_right.expanding_write(hann,error=False)

        return None, pyaudio.paContinue

    def non_blocking_stream_write(self, in_data, frame_count, time_info, status):
        if len(self.mask_buffer_left) < (18) or len(self.mask_buffer_right) < (18):
            audio = numpy.zeros((self._processing_size, self.channels), dtype=self.dtype)
            return audio, pyaudio.paContinue
        else:
            print(len(self.stft_buffer_left))
            chans = []
            # for each channel:
            mask = self.mask_buffer_left.read_overlap(18, 9)
            #this allows us to gather future, present, past
            #and then we crop it and apply it
            mask_working = numpy.vstack((self.mask_buffer_old_left,mask))
            mask_smoothed = smooth_mask(mask_working, self.residue, self.entropy_constant)  # remember smooth_mask also crops
            self.mask_buffer_old_left = mask[0:9,:]
            #save present for use on the next frame.
            #note: This is not assured to work properly.
            #for example: if audio skips, then the past buffer won't be a good mask candidate,
            #which will detrimentally impact smoothing and min/max, not to mention entropy.
            bins = self.stft_buffer_left.read(9)
            masked = bins# * mask_smoothed
            chans.append(self.stft_hann.synthesis(masked))

            mask = self.mask_buffer_right.read_overlap(18, 9)
            mask_working = numpy.vstack((self.mask_buffer_old_right, mask))
            mask_smoothed = smooth_mask(mask_working, self.residue, self.entropy_constant)  # remember smooth_mask also crops
            self.mask_buffer_old_right = mask[0:9,:] #write the old mask
            bins = self.stft_buffer_right.read(9)
            masked = bins #* mask_smoothed #remove this once debugged
            chans.append(self.stft_hann.synthesis(masked))

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
    SS = StreamSampler()
    SS.listen()


    def close():
        dpg.destroy_context()
        SS.stop()
        quit()


    dpg.create_context()

    dpg.create_viewport(title='Streamclean', height=100, width=400)
    dpg.setup_dearpygui()
    dpg.configure_app(auto_device=True)

    dpg.show_viewport()
    dpg.start_dearpygui()
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
    close()  # clean up the program runtime when the user closes the window
