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
#"realtime" ? 36 ms delay, and the occasional frame skip.


import numpy
from pyroomacoustics.transform.stft import compute_synthesis_window
from ssqueezepy import stft,istft
import pyaudio

from np_rw_buffer import RingBuffer

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
    for i in range(5):
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
    smoothed = numpy.convolve(entropy, numpy.ones(3), 'same') / 3
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


def man(arr):
    med = numpy.nanmedian(arr[numpy.nonzero(arr)])
    return numpy.nanmedian(numpy.abs(arr - med))

def atd(arr):
    x = numpy.square(numpy.abs(arr - man(arr)))
    return numpy.sqrt(numpy.nanmean(x))

def iterated_mask_generation(logit: numpy.ndarray, hann: numpy.ndarray, boxcar: numpy.ndarray, NBINS):
    mask = numpy.zeros((hann.shape[0], hann.shape[1] + 1))
    for each in range(hann.shape[0]):
        mask[each, -1] = atomic_entropy(logit, hann[each, :NBINS], boxcar[each, :NBINS])
    e = hann.copy()
    threshold = atd(e) + man(e)
    e[e < threshold] = 0
    e[e > 0] = 1
    mask[:,:-1] = e
    return mask  # this should return a 9x130 array with the entropy in the final row


def smooth_mask(mask: numpy.ndarray, residue_constant: float, entropy_constant: float):
    entropy = mask[:, -1]  # should be 27 samples
    entropy = entropy_gate_and_smoothing(entropy, entropy_constant)
    mask = mask[:, :-1]
    
    #disabling thresholding
    smoothed = numpy.ones(mask.shape) * entropy[:,None]

    #smoothed = numpyfilter_wrapper_50(smoothed)
    smoothed[smoothed < residue_constant] = residue_constant
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

def hann_local(M, sym=True):
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
        self.fft_len = 512
        self.hop_ratio = 4
        self.NBINS = 63  #around or less than 4000hz
        self.hop_size = self.fft_len // self.hop_ratio
        self.hann_window = hann_local(self.fft_len) #benefit of asymetric window not established. hann(self.fft_len, flag='asymmetric', length='full')
        self.boxcar_window = boxcar(self.fft_len)
        self.synthesis_window = compute_synthesis_window(self.hann_window, self.hop_size)
        self._processing_size = ((self.fft_len * 2) + self.hop_size)

        self.logit = generate_reasonable_logistic(self.NBINS)
        self._sample_rate = sample_rate
        self._channels = channels
        self.channels = channels
        self.residue = 0.00001
        self.entropy_constant = 0.065

        self.stft_buffer_left = RingBuffer(27, 257)  # we need 18 x 129
        self.stft_buffer_left.dtype = numpy.complex128
        self.smoothed_buffer_left = RingBuffer(27, 257)  # we need 18 x 129
        self.smoothed_buffer_left.dtype = numpy.complex128

        self.mask_buffer_left = RingBuffer(27, 258)
        self.mask_buffer_left.dtype = numpy.float64
        self.mask_buffer_old_left = numpy.ones((9,258),dtype=numpy.float64)

        self.stft_buffer_right = RingBuffer(27, 257)
        self.stft_buffer_right.dtype = numpy.complex128
        self.smoothed_buffer_right = RingBuffer(27, 257)
        self.smoothed_buffer_right.dtype = numpy.complex128

        self.mask_buffer_right = RingBuffer(27, 258)
        self.mask_buffer_right.dtype = numpy.float64
        self.mask_buffer_old_right = numpy.ones((9, 258), dtype=numpy.float64)


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


        boxcar = stft(x=audio_in_left, window=self.boxcar_window, n_fft=self.fft_len, hop_len=self.hop_size).T
        hann = stft(x=audio_in_left, window=self.hann_window, n_fft=self.fft_len, hop_len=self.hop_size).T
        mask = iterated_mask_generation(self.logit, numpy.abs(hann), numpy.abs(boxcar), self.NBINS)
        self.mask_buffer_left.expanding_write(mask,error=False)
        self.stft_buffer_left.expanding_write(hann,error=False)

        audio_in_right = audio_in[:, 1]
        boxcar =  stft(x=audio_in_right, window=self.boxcar_window, n_fft=self.fft_len, hop_len=self.hop_size).T
        hann = stft(x=audio_in_right, window=self.hann_window, n_fft=self.fft_len, hop_len=self.hop_size).T
        mask = iterated_mask_generation(self.logit, numpy.abs(hann), numpy.abs(boxcar), self.NBINS)
        self.mask_buffer_right.expanding_write(mask,error=False)
        self.stft_buffer_right.expanding_write(hann,error=False)

        if len (self.mask_buffer_left)> (17) and len(self.mask_buffer_right) > (17):
            mask = self.mask_buffer_left.read_overlap(18, 9)
            mask_working = numpy.vstack((self.mask_buffer_old_left,mask))
            mask_smoothed = smooth_mask(mask_working, self.residue, self.entropy_constant)
            self.mask_buffer_old_left = mask[0:9,:]
            bins = self.stft_buffer_left.read(9)
            masked = bins * mask_smoothed[9:18,:]
            self.smoothed_buffer_left.expanding_write(masked, error=False)

            mask = self.mask_buffer_right.read_overlap(18, 9)
            mask_working = numpy.vstack((self.mask_buffer_old_right,mask))
            mask_smoothed = smooth_mask(mask_working, self.residue,
                                    self.entropy_constant)  # remember smooth_mask also crops

            self.mask_buffer_old_right = mask[0:9, :]
            bins = self.stft_buffer_right.read(9)

            masked = bins * mask_smoothed[9:18,:]
            self.smoothed_buffer_right.expanding_write(masked, error=False)


        return None, pyaudio.paContinue

    def non_blocking_stream_write(self, in_data, frame_count, time_info, status):
        if len(self.smoothed_buffer_right) < (9) or len(self.smoothed_buffer_left) < (9):
            audio = numpy.zeros((self._processing_size, self.channels), dtype=self.dtype)
            return audio, pyaudio.paContinue
        else:
            chans = []
            bins = self.smoothed_buffer_left.read(9).T
            chans.append(istft(Sx=bins, window=self.synthesis_window, n_fft=self.fft_len, hop_len=self.hop_size, N=self._processing_size))

            bins = self.smoothed_buffer_right.read(9).T
            chans.append(istft(Sx=bins, window=self.synthesis_window, n_fft=self.fft_len, hop_len=self.hop_size, N=self._processing_size))
            #if we do absolutely nothing here, there's no risk of loosing sync
            #so all we do here is monitor for available audio and send it out

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
        SS.stop()
        quit()

    while SS.micstream.is_active():
        eval(input("press any key to quit\n"))
