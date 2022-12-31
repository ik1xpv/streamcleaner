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

# realtime_entropy
#well, it's not going to mask, but it's got entropy


import numpy
import numba
from pyroomacoustics.transform.stft import compute_synthesis_window
from ssqueezepy import stft,istft
import pyaudio
import dearpygui.dearpygui as dpg

logit_window = numpy.asarray([0.,0.19950109,0.23733074,0.25954935,0.275377,0.28770315,0.29781484,0.30639864,0.31386436,0.32047634,0.32641507,0.3318093,0.33675408,0.34132161,0.34556799,0.34953773,0.35326668,0.35678419,0.36011454,0.36327811,0.36629212,0.36917128,0.37192823,0.37457394,0.37711797,0.37956873,0.38193361,0.38421919,0.38643132,0.38857524,0.39065566,0.39267682,0.39464256,0.39655637,0.39842143,0.40024063,0.40201664,0.40375188,0.40544862,0.40710891,0.40873468,0.41032769,0.41188959,0.41342192,0.41492609,0.41640345,0.41785523,0.4192826,0.42068666,0.42206842,0.42342885,0.42476887,0.42608931,0.42739101,0.42867471,0.42994114,0.43119098,0.43242489,0.43364348,0.43484733,0.436037,0.43721302,0.43837589,0.43952611,0.44066412,0.44179037,0.44290528,0.44400924,0.44510265,0.44618588,0.44725927,0.44832318,0.44937792,0.45042381,0.45146115,0.45249024,0.45351136,0.45452478,0.45553075,0.45652954,0.45752138,0.45850651,0.45948517,0.46045756,0.46142392,0.46238443,0.46333932,0.46428876,0.46523296,0.46617209,0.46710635,0.46803589,0.46896091,0.46988155,0.47079798,0.47171037,0.47261886,0.47352361,0.47442477,0.47532247,0.47621687,0.4771081,0.4779963,0.47888161,0.47976414,0.48064404,0.48152143,0.48239643,0.48326917,0.48413976,0.48500834,0.48587501,0.4867399,0.48760311,0.48846476,0.48932497,0.49018384,0.49104149,0.49189802,0.49275355,0.49360817,0.49446201,0.49531517,0.49616774,0.49701984,0.49787158,0.49872305,0.49957437,0.50042563,0.50127695,0.50212842,0.50298016,0.50383226,0.50468483,0.50553799,0.50639183,0.50724645,0.50810198,0.50895851,0.50981616,0.51067503,0.51153524,0.51239689,0.5132601,0.51412499,0.51499166,0.51586024,0.51673083,0.51760357,0.51847857,0.51935596,0.52023586,0.52111839,0.5220037,0.5228919,0.52378313,0.52467753,0.52557523,0.52647639,0.52738114,0.52828963,0.52920202,0.53011845,0.53103909,0.53196411,0.53289365,0.53382791,0.53476704,0.53571124,0.53666068,0.53761557,0.53857608,0.53954244,0.54051483,0.54149349,0.54247862,0.54347046,0.54446925,0.54547522,0.54648864,0.54750976,0.54853885,0.54957619,0.55062208,0.55167682,0.55274073,0.55381412,0.55489735,0.55599076,0.55709472,0.55820963,0.55933588,0.56047389,0.56162411,0.56278698,0.563963,0.56515267,0.56635652,0.56757511,0.56880902,0.57005886,0.57132529,0.57260899,0.57391069,0.57523113,0.57657115,0.57793158,0.57931334,0.5807174,0.58214477,0.58359655,0.58507391,0.58657808,0.58811041,0.58967231,0.59126532,0.59289109,0.59455138,0.59624812,0.59798336,0.59975937,0.60157857,0.60344363,0.60535744,0.60732318,0.60934434,0.61142476,0.61356868,0.61578081,0.61806639,0.62043127,0.62288203,0.62542606,0.62807177,0.63082872,0.63370788,0.63672189,0.63988546,0.64321581,0.64673332,0.65046227,0.65443201,0.65867839,0.66324592,0.6681907,0.67358493,0.67952366,0.68613564,0.69360136,0.70218516,0.71229685,0.724623,0.74045065,0.76266926,0.80049891,1.,1.,0.80049891,0.76266926,0.74045065,0.724623,0.71229685,0.70218516,0.69360136,0.68613564,0.67952366,0.67358493,0.6681907,0.66324592,0.65867839,0.65443201,0.65046227,0.64673332,0.64321581,0.63988546,0.63672189,0.63370788,0.63082872,0.62807177,0.62542606,0.62288203,0.62043127,0.61806639,0.61578081,0.61356868,0.61142476,0.60934434,0.60732318,0.60535744,0.60344363,0.60157857,0.59975937,0.59798336,0.59624812,0.59455138,0.59289109,0.59126532,0.58967231,0.58811041,0.58657808,0.58507391,0.58359655,0.58214477,0.5807174,0.57931334,0.57793158,0.57657115,0.57523113,0.57391069,0.57260899,0.57132529,0.57005886,0.56880902,0.56757511,0.56635652,0.56515267,0.563963,0.56278698,0.56162411,0.56047389,0.55933588,0.55820963,0.55709472,0.55599076,0.55489735,0.55381412,0.55274073,0.55167682,0.55062208,0.54957619,0.54853885,0.54750976,0.54648864,0.54547522,0.54446925,0.54347046,0.54247862,0.54149349,0.54051483,0.53954244,0.53857608,0.53761557,0.53666068,0.53571124,0.53476704,0.53382791,0.53289365,0.53196411,0.53103909,0.53011845,0.52920202,0.52828963,0.52738114,0.52647639,0.52557523,0.52467753,0.52378313,0.5228919,0.5220037,0.52111839,0.52023586,0.51935596,0.51847857,0.51760357,0.51673083,0.51586024,0.51499166,0.51412499,0.5132601,0.51239689,0.51153524,0.51067503,0.50981616,0.50895851,0.50810198,0.50724645,0.50639183,0.50553799,0.50468483,0.50383226,0.50298016,0.50212842,0.50127695,0.50042563,0.49957437,0.49872305,0.49787158,0.49701984,0.49616774,0.49531517,0.49446201,0.49360817,0.49275355,0.49189802,0.49104149,0.49018384,0.48932497,0.48846476,0.48760311,0.4867399,0.48587501,0.48500834,0.48413976,0.48326917,0.48239643,0.48152143,0.48064404,0.47976414,0.47888161,0.4779963,0.4771081,0.47621687,0.47532247,0.47442477,0.47352361,0.47261886,0.47171037,0.47079798,0.46988155,0.46896091,0.46803589,0.46710635,0.46617209,0.46523296,0.46428876,0.46333932,0.46238443,0.46142392,0.46045756,0.45948517,0.45850651,0.45752138,0.45652954,0.45553075,0.45452478,0.45351136,0.45249024,0.45146115,0.45042381,0.44937792,0.44832318,0.44725927,0.44618588,0.44510265,0.44400924,0.44290528,0.44179037,0.44066412,0.43952611,0.43837589,0.43721302,0.436037,0.43484733,0.43364348,0.43242489,0.43119098,0.42994114,0.42867471,0.42739101,0.42608931,0.42476887,0.42342885,0.42206842,0.42068666,0.4192826,0.41785523,0.41640345,0.41492609,0.41342192,0.41188959,0.41032769,0.40873468,0.40710891,0.40544862,0.40375188,0.40201664,0.40024063,0.39842143,0.39655637,0.39464256,0.39267682,0.39065566,0.38857524,0.38643132,0.38421919,0.38193361,0.37956873,0.37711797,0.37457394,0.37192823,0.36917128,0.36629212,0.36327811,0.36011454,0.35678419,0.35326668,0.34953773,0.34556799,0.34132161,0.33675408,0.3318093,0.32641507,0.32047634,0.31386436,0.30639864,0.29781484,0.28770315,0.275377,0.25954935,0.23733074,0.19950109,0.])


from np_rw_buffer import RingBuffer

@numba.jit()
def atomic_entropy(logit: numpy.ndarray, hann: numpy.ndarray):
    hann = numpy.sort(hann)
    hann = numpy.interp(hann, (hann[0], hann[-1]), (0, +1))
    return (1 - numpy.corrcoef(hann, logit)[0, 1])
    # returns the pre-smoothed, averaged individual time bin entropy calculation

@numba.jit()
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


def smooth_mask(mask_working, residue, entropy_c):
    flat = mask_working.flatten()
    mask_working = numpy.squeeze(flat)
    smoothed = numpy.convolve(mask_working, numpy.ones(14), 'same') / 14
    smoothed_gated = smoothed.copy()

    smoothed_gated[smoothed_gated < entropy_c] = 0
    smoothed_gated[smoothed_gated > 0] = 1
    streak = longestConsecutive(smoothed_gated)
    total = numpy.sum(smoothed_gated)
    # required total = 14
    # required streak = 11
    if total < 22 and streak < 16:  # there is probabilistically no signals present here.
        return numpy.ones((37+74,257),dtype=numpy.float64) * residue
    else:
        mask = numpy.ones((37+74,257),dtype=numpy.float64)
        mask = mask * smoothed_gated[:,None]
        smoothed = numpyfilter_wrapper_50(mask)
        smoothed = (smoothed - numpy.nanmin(smoothed)) / numpy.ptp(smoothed)
        smoothed[smoothed < residue] = residue
        return smoothed

@numba.jit()
def generate_reasonable_logistic(points):
    fprint = numpy.linspace(0.0, 1.0, points)
    fprint[1:-1] /= 1 - fprint[1:-1]
    fprint[1:-1] = numpy.log(fprint[1:-1])
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

    def __init__(self, sample_rate=48000, channels=2,  micindex=1, speakerindex=1, dtype=numpy.float32):
        self.pa = pyaudio.PyAudio()
        self.fft_len = 512
        self.hop_ratio = 4
        self.NBINS = 32  #around or less than 4000hz
        self.hop_size = self.fft_len // self.hop_ratio
        self.hann_window = hann_local(self.fft_len) #benefit of asymetric window not established. hann(self.fft_len, flag='asymmetric', length='full')
        self.synthesis_window = compute_synthesis_window(self.hann_window, self.hop_size)
        self._processing_size = 4736 #roughly 37 bins, or a little less than 99 ms

        self._sample_rate = sample_rate
        self._channels = channels
        self.channels = channels
        self.residue = 0.0001
        self.entropy_constant = 0.0204

        self.stft_buffer_left = RingBuffer(370, 257)  #need a total of 200
        self.stft_buffer_left.dtype = numpy.complex128

        self.mask_buffer_left = RingBuffer(370,1)
        self.mask_buffer_left.dtype = numpy.float64
        self.mask_buffer_old_left = numpy.ones((37,1),dtype=numpy.float64)

        self.stft_buffer_right = RingBuffer(370, 257)
        self.stft_buffer_right.dtype = numpy.complex128

        self.mask_buffer_right = RingBuffer(370,1)
        self.mask_buffer_right.dtype = numpy.float64
        self.mask_buffer_old_right = numpy.ones((37,1), dtype=numpy.float64)


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
        bins = self.NBINS
        logit = generate_reasonable_logistic(bins)

        logos = numpy.abs(stft(x= audio_in[:, 0], window=logit_window, n_fft=self.fft_len, hop_len=self.hop_size).T)
        hann = stft(x= audio_in[:, 0], window=self.hann_window, n_fft=self.fft_len, hop_len=self.hop_size).T

        entropy = numpy.zeros(37)

        for each in range(37):
            entropy[each] = atomic_entropy(logit, logos[each,:bins])

        self.mask_buffer_left.expanding_write(entropy,error=False)

        self.stft_buffer_left.expanding_write(hann,error=False)

        logos = numpy.abs(stft(x= audio_in[:, 1], window=logit_window, n_fft=self.fft_len, hop_len=self.hop_size).T)
        hann = stft(x= audio_in[:, 1], window=self.hann_window, n_fft=self.fft_len, hop_len=self.hop_size).T

        entropy = numpy.zeros(37)

        for each in range(37):
            entropy[each] = atomic_entropy(logit, logos[each, :bins])

        self.mask_buffer_right.expanding_write(entropy, error=False)

        self.stft_buffer_right.expanding_write(hann,error=False)

        return None, pyaudio.paContinue

    def non_blocking_stream_write(self, in_data, frame_count, time_info, status):

        if len(self.mask_buffer_left) < (74) or len(self.mask_buffer_right) < (74):
            audio = numpy.zeros((self._processing_size, self.channels), dtype=self.dtype)
            return audio, pyaudio.paContinue
        else:
            entropy_c = self.entropy_constant
            chans = []
            mask = self.mask_buffer_left.read_overlap(74, 37)

            mask_working = numpy.vstack((self.mask_buffer_old_left,mask))
            mask_smoothed = smooth_mask(mask_working, self.residue, entropy_c)
            self.mask_buffer_old_left = mask[0:37,:]

            bins = self.stft_buffer_left.read(37)
            masked = bins * mask_smoothed[37:74,:]
            masked = masked[:37,:].T
            chans.append(istft(Sx=masked, window=self.hann_window, n_fft=self.fft_len, hop_len=self.hop_size, N=self._processing_size))

            mask = self.mask_buffer_right.read_overlap(74, 37)
            mask_working = numpy.vstack((self.mask_buffer_old_right,mask))
            mask_smoothed = smooth_mask(mask_working, self.residue,
                                    entropy_c)  # remember smooth_mask also crops

            self.mask_buffer_old_right = mask[0:37,:]
            bins = self.stft_buffer_right.read(37)

            masked = bins *  mask_smoothed[37:74,:]
            masked = masked[:37,:].T
            chans.append(istft(Sx=masked, window=self.hann_window, n_fft=self.fft_len, hop_len=self.hop_size,N=self._processing_size))
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
    def entropy_slider_function():
        SS.entropy_constant =  dpg.get_value("entropy")

    def nbins_slider_function():
        SS.NBINS = dpg.get_value("nbins")

    dpg.create_context()
    with dpg.window(tag="Primary Window", height=30, width=100):
        dpg.add_slider_float(tag="entropy", label="entropy", min_value=0.010,max_value=0.50,default_value=0.050, callback= entropy_slider_function)
        dpg.add_slider_int(tag="nbins", label="nbins",min_value=16,max_value=257,default_value=32 ,callback = nbins_slider_function)

    dpg.create_viewport(title= 'Streamclean', height=30, width=100)
    dpg.setup_dearpygui()
    dpg.configure_app(auto_device=True)
    dpg.show_viewport()
    dpg.set_primary_window("Primary Window", True)
    dpg.start_dearpygui()
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
    close()  # clean up the program runtime when the user closes the window

