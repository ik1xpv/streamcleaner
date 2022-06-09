import numpy
import numba
import pyaudio
from np_rw_buffer import AudioFramingBuffer
from threading import Thread
import dearpygui.dearpygui as dpg
from pywt import dwtn
from time import sleep
import planfftw
import noisereduce
EPS = 1e-8
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)


def segm_tec(f, N):
    x = numpy.asarray(f)
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = numpy.where(numpy.isnan(x))[0]
    indl = numpy.asarray(indnan)

    if indl.size != 0:
        x[indnan] = numpy.inf
        dx[numpy.where(numpy.isnan(dx))[0]] = numpy.inf

    vil = numpy.zeros(dx.size + 1)
    vil[:-1] = dx[:]
    vix = numpy.zeros(dx.size + 1)
    vix[1:] = dx[:]

    ind = numpy.unique(numpy.where((vil > 0) & (vix <= 0))[0])
    if ind.size < 2:
        return numpy.asarray([0,len(f)])
    locmax = numpy.zeros((f.size),dtype=numpy.float64)
    locmax[ind] = f[ind]
    locmax[0] = 0
    locmax[-1] = 0 #ends cannot be peaks!
    desc_sort_index = numpy.argsort(locmax)[::-1]
    desc_sort_index = desc_sort_index[0:ind.size]
    if N != 0:  # keep the N-th highest maxima and their index
        if len(desc_sort_index) > N:
            desc_sort_index = desc_sort_index[0:N + 1]
        else:
            N = desc_sort_index.size
        desc_sort_index = numpy.sort(desc_sort_index)  # gotta sort them again
        bounds = numpy.zeros((N + 2), dtype=numpy.int64)
        bounds[1] = (numpy.argmin(f[0:desc_sort_index[0]]))  # -2
        for i in range(N - 2):
            bounds[i + 2] = (desc_sort_index[i] + numpy.argmin(f[desc_sort_index[i]:desc_sort_index[i + 1]]) - 1)
        bounds[-2] = (desc_sort_index[N] + numpy.argmin(f[desc_sort_index[N]:len(f)]) - 1)
        bounds[-1] = f.size
    return numpy.asarray(bounds)


def EFD(x: numpy.ndarray, N: int,bounds: numpy.ndarray):
    if N < 1:
        return x

    # we will now implement the Empirical Fourier Decomposition
    x = numpy.asarray(x, dtype=numpy.float64)
    fa = planfftw.rfft(x.shape)
    ff = fa(x)
    # extract the boundaries of Fourier segments
    #with numba.objmode(out='int[:]'):
    #if bounds == None:
    #bounds = segm_tec(numpy.absolute(ff[0:round(ff.size / 2)]), N)
    if bounds.size < 3:
        return x.astype(dtype=numpy.float32) #no need to go further because we have nothing to work with
    # truncate the boundaries to [0,pi]
    bounds = bounds * numpy.pi / round(len(ff) / 2)

    # extend the signal by miroring to deal with the boundaries
    l = round(len(x) / 2)
    z = numpy.lib.pad(x, ((round(len(x) / 2)), round(len(x) / 2)), 'symmetric')
    fz = planfftw.rfft(z.shape)

    ff = fz(z)
    # obtain the boundaries in the extend f
    bound2 = numpy.ceil(bounds * round(len(ff) / 2) / numpy.pi).astype(dtype=numpy.int64)
    efd = numpy.zeros(((len(bound2) - 1, len(x))), dtype=numpy.float64)
    ft = numpy.zeros((efd.shape[0], len(ff)), dtype=numpy.cdouble)
    fx = planfftw.irfft(len(ff))
    # define an ideal functions and extract components
    for k in range(efd.shape[0]):
        if bound2[k] == 0:
            ft[k, 0:bound2[k + 1]] = ff[0:bound2[k + 1]]
            # ft[k,len(ff)+1-bound2[k+1]:len(ff)] = ff[len(ff)+1-bound2[k+1]:len(ff)]
            ft[k, -bound2[k + 1]:len(ff)] = ff[-bound2[k + 1]:len(ff)]

        else:
            ft[k, bound2[k]:bound2[k + 1]] = ff[bound2[k]:bound2[k + 1]]
            # ft[k,len(ff)+1-bound2[k+1]:len(ff)+1-bound2[k]] = ff[len(ff)+1-bound2[k+1]:len(ff)+1-bound2[k]]
            ft[k, -bound2[k + 1]:-bound2[k]] = ff[-bound2[k + 1]:-bound2[k]]
        rx = fx(ft[k, :])
        efd[k, :] = rx[l:-l].real

    return efd.astype(dtype=numpy.float32)


@numba.jit(numba.float32[:](numba.float32[:]),nogil=True, cache=True,fastmath=True)
def average_neigbors(data: numpy.ndarray):
    output = numpy.zeros_like(data,dtype=numpy.float32)
    for i in numba.prange(data.size - 2):
        output[i+1] = (data[i] + data[i+1] + data[i+2]) / 3
    output[0] = (data[0] + (data[1] + data[-2]) / 2) / 3
    output[-1] = (data[-1] + (data[-2] + data[-3]) / 2) / 3
    return output

@numba.jit(numba.float32[:](numba.float32[:],numba.float32),nogil=True, cache=True,fastmath=True)
def numba_fabada(data: numpy.ndarray,sigma: numpy.float32):
    x = numpy.zeros_like(data,dtype=numpy.float32)
    x[:] = data.copy()
    x[numpy.where(numpy.isnan(data))] =  2.2250738585072014e-308
    z = x.copy()
    iterations: int = 1
    N = x.size
    max_iterations = 15

    bayesian_weight = numpy.zeros_like(x,dtype=numpy.float32)
    bayesian_model = numpy.zeros_like(x,dtype=numpy.float32)
    model_weight = numpy.zeros_like(x,dtype=numpy.float32)

    # pre-declaring all arrays allows their memory to be allocated in advance
    posterior_mean = numpy.zeros_like(x,dtype=numpy.float32)
    posterior_mean[:] = x.copy()

    initial_evidence = numpy.zeros_like(x,dtype=numpy.float32)
    evidence = numpy.zeros_like(x,dtype=numpy.float32)
    prior_mean = numpy.zeros_like(x,dtype=numpy.float32)
    prior_variance = numpy.zeros_like(x,dtype=numpy.float32)
    posterior_variance = numpy.zeros_like(x,dtype=numpy.float32)

    chi2_data_min = N
    data_variance = numpy.zeros_like(x,dtype=numpy.float32)
    if sigma == 0:
        return x #we are done here!
    data_variance.fill(sigma**2)
    posterior_variance[:] = data_variance.copy()
    prior_variance[:] = data_variance.copy()

    prior_mean[:] = x.copy()

    # fabada figure 14
    #formula 3, but for initial assumptions

    upper = numpy.square(-sigma)
    lower = 2 * sigma**2

    first = (-upper / lower)
    second = numpy.sqrt(2 * numpy.pi) * sigma**2
    evidence[:] = numpy.exp(first) / second
    initial_evidence[:] = evidence.copy()
    evidence_previous = numpy.mean(evidence)

    while 1:

        # GENERATES PRIORS

        prior_mean[:] = average_neigbors(posterior_mean)

        prior_variance = posterior_variance.copy() #if this is an array, you must use .copy() or it will
        #cause any changes made to posterior_variance to also automatically be applied to prior
        #variance, making these variables redundant.


        # APPLY BAYES' THEOREM
        # fabada figure 8?
        for i in numba.prange(N):
                posterior_variance[i] = (data_variance[i] * prior_variance[i])/(data_variance[i] + prior_variance[i])

        # fabada figure 7
        for i in numba.prange(N):
            posterior_mean[i] = (
                ((prior_mean[i] / prior_variance[i]) + (x[i] / data_variance[i])) * posterior_variance[i])

        upper = numpy.square(prior_mean - x)
        lower = 2 * (prior_variance + data_variance)
        first = (-upper/lower)
        second = numpy.sqrt(2*numpy.pi) * prior_variance + data_variance

        evidence = numpy.exp(first) / second

        # fabada figure 6: probability distribution calculation
        evidence_derivative = numpy.mean(evidence) - evidence_previous
        evidence_previous = numpy.mean(evidence)

        # EVALUATE CHI2
        chi2_data = numpy.sum((x - posterior_mean) ** 2 / (data_variance))

        if iterations == 1:
            chi2_data_min = chi2_data

        # COMBINE MODELS FOR THE ESTIMATION

        for i in numba.prange(N):
            model_weight[i] = evidence[i] * chi2_data

        for i in numba.prange(N):
            bayesian_weight[i] = bayesian_weight[i] + model_weight[i]
            bayesian_model[i] = bayesian_model[i] + (model_weight[i] * posterior_mean[i])

        if ((chi2_data > N) and (evidence_derivative < 0)) \
                or (iterations > max_iterations):  # don't overfit the data
            break
        iterations = iterations + 1
        # COMBINE ITERATION ZERO
    for i in numba.prange(N):
        model_weight[i] = initial_evidence[i] * chi2_data_min
    for i in numba.prange(N):
        bayesian_weight[i] = bayesian_weight[i] + model_weight[i]
        bayesian_model[i] = bayesian_model[i] + (model_weight[i] * x[i])

    for i in numba.prange(N):
                if bayesian_weight[i] != 0:
                    x[i] = bayesian_model[i] / bayesian_weight[i]
                else:
                    x[i] = bayesian_model[i]
    for i in numba.prange(N):
            if numpy.isnan(x[i]):
                x[i] = z[i]  # don't return NaN values thank you
    return x

def signaltonoise(a, axis=0, ddof=0):
    a = numpy.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return numpy.where(sd == 0, 0, m/sd)

class FilterRun(Thread):
    def __init__(self, rb, pb, channels, processing_size, dtype):
        super(FilterRun, self).__init__()
        self.running = True
        self.rb = rb
        self.processedrb = pb
        self.channels = channels
        self.processing_size = processing_size
        self.dtype = dtype
        self.buffer = numpy.ndarray(dtype=numpy.float32, shape=[int(self.processing_size * self.channels)])
        self.buffer2 = numpy.ndarray(dtype=numpy.float32, shape=[int(self.processing_size * self.channels)])
        self.buffer = self.buffer.reshape(-1, self.channels)
        self.buffer2 = self.buffer.reshape(-1, self.channels)

    def write_filtered_data(self):

        numpy.copyto(self.buffer, self.rb.read(self.processing_size).astype(dtype=numpy.float32))
        for i in range(self.channels):
            workload = self.buffer[:,i]
            coeffs = dwtn(workload, wavelet='db2')
            detail_coeffs = coeffs['d' * workload.ndim]
            detail_coeffs = detail_coeffs[numpy.nonzero(detail_coeffs)]
            sigma = numpy.median(numpy.abs(detail_coeffs)) / 0.6616518484657332
            workload[numpy.isnan(workload)] = 0  # replace NaN with zeros
            filtered = noisereduce.reduce_noise(y=workload, sr=44100)  # isolate the signal if possible
            residual = workload - filtered
            fa = planfftw.rfft(workload.shape)
            ff = fa(workload)
            # extract the boundaries of the Fourier segments which are most significant
            # with numba.objmode(out='int[:]'):
            bounds = segm_tec(numpy.absolute(ff[0:round(ff.size / 2)]), 14) #let's assume there are around 14 maxima
            efd = EFD(residual,bounds.size-1,bounds)
            if efd.ndim > 1:
                efd[-1,:] = 0 #erase the last EFD. we don't need this.
                residual = numpy.sum(efd, axis=0) #reduce the array back to the original shape
            else:
                residual = efd # we can't do anything here
            smoothed = numba_fabada(residual,sigma) #minimize the noise by statistical analysis
            prepared = smoothed + filtered # restore the original waveform with statistically determined noise reduction applied to residuals
            prepared[numpy.isnan(prepared)] = 0 #replace NaN with zeros
            product =  noisereduce.reduce_noise(y=prepared, sr=44100) #gate again with maximized isolation
            efd1 = EFD(product, bounds.size-1, bounds)
            if efd1.ndim > 1:
                efd1[-1, :] = 0  # erase the last EFD. we don't need this.
                product = numpy.sum(efd1, axis=0)  # reduce the array back to the original shape
            else:
                product = efd1  # we can't do anything here
            self.buffer2[:, i] = product
        #indemnible = self.buffer2.transpose(1,0)
        #result = numpy.mean(indemnible, axis=tuple(range(indemnible.ndim - 1)))
        #import time
        #start_time = time.time()
        #processed = self.vf.restore_inmem(result)
        #print("--- %s seconds ---" % (time.time() - start_time))
        #self.buffer2[:, 1] = processed #cross your fingers and hope this works
        #self.buffer2[:, 0] = processed #cross your fingers and hope this works

        self.processedrb.growing_write(self.buffer2.astype(dtype=self.dtype))

    def run(self):
        while self.running:
            if len(self.rb) < self.processing_size:
                sleep(0.01)  # idk how long we should sleep
            else:
                self.write_filtered_data()

    def stop(self):
        self.running = False


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

    def __init__(self, sample_rate=44100, channels=2, buffer_delay=1.5,  # or 1.5, measured in seconds
                 micindex=1, speakerindex=1, dtype=numpy.float32):
        self.pa = pyaudio.PyAudio()
        self._processing_size = sample_rate
        # np_rw_buffer (AudioFramingBuffer offers a delay time)
        self._sample_rate = sample_rate
        self._channels = channels
        self.ticker = 0
        self.rb = AudioFramingBuffer(sample_rate=sample_rate, channels=channels,
                                     seconds=6,  # Buffer size (need larger than processing size)[seconds * sample_rate]
                                     buffer_delay=0,  # #this buffer doesnt need to have a size
                                     dtype=numpy.dtype(dtype))

        self.processedrb = AudioFramingBuffer(sample_rate=sample_rate, channels=channels,
                                              seconds=6,
                                              # Buffer size (need larger than processing size)[seconds * sample_rate]
                                              buffer_delay=0,
                                              # as long as fabada completes in O(n) of less than the sample size in time
                                              dtype=numpy.dtype(dtype))
        self.filterthread = FilterRun(self.rb, self.processedrb, self._channels, self._processing_size, self.dtype)
        self.micindex = micindex
        self.speakerindex = speakerindex
        self.micstream = None
        self.speakerstream = None
        self.speakerdevice = ""
        self.micdevice = ""

        # Set inputs for inheritance
        self.set_sample_rate(sample_rate)
        self.set_channels(channels)
        self.set_dtype(dtype)

    @property
    def processing_size(self):
        return self._processing_size

    @processing_size.setter
    def processing_size(self, value):
        self._processing_size = value
        self._update_streams()

    def get_sample_rate(self):
        return self._sample_rate

    def set_sample_rate(self, value):
        self._sample_rate = value
        try:  # RingBuffer
            self.rb.maxsize = int(value * 5)
            self.processedrb.maxsize = int(value * 5)
        except AttributeError:
            pass
        try:  # AudioFramingBuffer
            self.rb.sample_rate = value
            self.processedrb.sample_rate = value
        except AttributeError:
            pass
        self._update_streams()

    sample_rate = property(get_sample_rate, set_sample_rate)

    def get_channels(self):
        return self._channels

    def set_channels(self, value):
        self._channels = value
        try:  # RingBuffer
            self.rb.columns = value
            self.processedrb.columns = value
        except AttributeError:
            pass
        try:  # AudioFrammingBuffer
            self.rb.channels = value
            self.processedrb.channels = value
        except AttributeError:
            pass
        self._update_streams()

    channels = property(get_channels, set_channels)

    def get_dtype(self):
        return self.rb.dtype

    def set_dtype(self, value):
        try:
            self.rb.dtype = value
        except AttributeError:
            pass
        self._update_streams()

    dtype = property(get_dtype, set_dtype)

    @property
    def pa_format(self):
        return self.get_pa_format(self.dtype)

    @pa_format.setter
    def pa_format(self, value):
        for np_dtype, pa_fmt in self.dtype_to_paformat.items():
            if value == pa_fmt:
                self.dtype = numpy.dtype(np_dtype)
                return

        raise ValueError('Invalid pyaudio format given!')

    @property
    def buffer_delay(self):
        try:
            return self.rb.buffer_delay
        except (AttributeError, Exception):
            return 0

    @buffer_delay.setter
    def buffer_delay(self, value):
        try:
            self.rb.buffer_delay = value
            self.processedrb.buffer_delay = value
        except AttributeError:
            pass

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
        try:
            self.filterthread.join()
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

        stream = self.pa.open(format=self.pa_format,
                              channels=self.channels,
                              rate=int(self.sample_rate),
                              input=True,
                              input_device_index=self.micindex,  # device_index,
                              # each frame carries twice the data of the frames
                              frames_per_buffer=int(self._processing_size),
                              stream_callback=self.non_blocking_stream_read,
                              start=False  # Need start to be False if you don't want this to start right away
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

        stream = self.pa.open(format=self.pa_format,
                              channels=self.channels,
                              rate=int(self.sample_rate),
                              output=True,
                              output_device_index=self.speakerindex,
                              frames_per_buffer=int(self._processing_size),
                              stream_callback=self.non_blocking_stream_write,
                              start=False  # Need start to be False if you don't want this to start right away
                              )
        return stream

    # it is critical that this function do as little as possible, as fast as possible. numpy.ndarray is the fastest we can move.
    # attention: numpy.ndarray is actually faster than frombuffer for known buffer sizes
    def non_blocking_stream_read(self, in_data, frame_count, time_info, status):
        audio_in = numpy.ndarray(buffer=in_data, dtype=self.dtype,
                                            shape=[int(self._processing_size * self._channels)]).reshape(-1,
                                                                                                         self.channels)
        self.rb.write(audio_in, error=False)
        return None, pyaudio.paContinue

    def non_blocking_stream_write(self, in_data, frame_count, time_info, status):
        # Read raw data
        # filtered = self.rb.read(frame_count)
        # if len(filtered) < frame_count:
        #     filtered = numpy.zeros((frame_count, self.channels), dtype=self.dtype)
        if len(self.processedrb) < self.processing_size:
            # print('Not enough data to play! Increase the buffer_delay')
            # uncomment this for debug
            audio = numpy.zeros((self.processing_size, self.channels), dtype=self.dtype)
            return audio, pyaudio.paContinue

        audio = self.processedrb.read(self.processing_size)
        chans = []
        for i in range(self.channels):
            filtered = audio[:, i]
            chans.append(filtered)

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
    # after importing numpy, reset the CPU affinity of the parent process so
    # that it will use all cores
    SS = StreamSampler(buffer_delay=0)
    SS.listen()
    SS.filterthread.start()


    def close():
        dpg.destroy_context()
        SS.filterthread.stop()
        SS.stop()
        quit()


    dpg.create_context()
    dpg.create_viewport(title= 'Streamclean', height=100, width=400)
    dpg.setup_dearpygui()
    dpg.configure_app(auto_device=True)

    dpg.show_viewport()
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
    close()  # clean up the program runtime when the user closes the window
