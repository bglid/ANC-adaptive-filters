import math

import numpy as np
from numpy.random import sample
from numpy.typing import NDArray
from scipy.signal import get_window


class STFT:
    def __init__(self, window_length=512, window_type="hann"):
        self.window_length = window_length
        self.hop_size = window_length // 2
        # creating hann window:
        self.window = get_window(window_type, window_length)

    def dft_matrix(self):
        # to represent each samples count
        n = np.arange(self.window_length)
        # to get the frequency value at each sample
        f = np.arange(self.window_length).reshape((self.window_length, 1))
        return np.exp(-1j * (2 * np.pi * f * n / self.window_length))

    # inverse dft function
    def idft_matrix(self):
        n = np.arange(self.window_length)
        f = n.reshape((self.window_length, 1))
        exponent = 1j * (2.0 * np.pi * (n / self.window_length) * f)
        # to make this do the inverse DFT:
        return (1 / self.window_length) * (np.exp(exponent))

    # defining a function for data matrix prep
    def stft(self, signal: NDArray[np.float64]) -> NDArray[np.complex128]:
        n = len(signal)
        # Getting the padding size
        padding_diff = int(
            math.pow(2, math.ceil(math.log(self.window_length) / math.log(2)))
            - self.window_length
        )
        # getting the number of total frames
        n_frames = 1 + (n - self.window_length) // self.hop_size
        frames = np.empty((n_frames, self.window_length // 2 + 1), dtype=np.complex128)
        # Creating an array to hold the results per fft
        # frames = np.zeros(
        #     (1 + (self.window_length + padding_diff) // 2, n_frames),
        #     dtype=np.complex128,
        # )

        for j in range(n_frames):
            current_frame = j * self.hop_size
            # frame_j = signal[current_frame : current_frame + self.window_length]
            windowed_frame = (
                signal[current_frame : current_frame + self.window_length] * self.window
            )
            # padding the window
            padded_window = np.pad(windowed_frame, (0, padding_diff), mode="constant")

            # full_ftt = self.fft(padded_window)
            # frames[j] = full_ftt[: len(full_ftt) // 2 + 1]

            # np version below:
            frames[j] = np.fft.rfft(padded_window)

            # testing
            # np.testing.assert_allclose(
            #     results[:, j], np.fft.rfft(padded_window), rtol=1e-8, atol=1e-10
            # )
        return frames

    # to inverse this, we basically want the same procedure, but going in the other direction
    def istft(self, frames: NDArray[np.complex128]) -> NDArray[np.float64]:
        n_frames, n_sample_length = frames.shape
        # assert n_sample_length == self.window_length // 2 + 1

        # the output length should be the frames * N/2
        output_length = (n_frames - 1) * self.hop_size + self.window_length
        overlapped_row = np.zeros(output_length)
        # adding a weight matrix to divide by to handle overlapping samples
        weight_matrix = np.zeros(output_length)
        # setting a loop until we reach the last possible frame
        for index, freq in enumerate(frames):
            # adjusting window size to reflect going in opposite direction
            window = np.fft.irfft(freq, n=self.window_length)
            current_frame = index * self.hop_size
            overlapped_row[current_frame : current_frame + self.window_length] += (
                window * self.window
            )
            # updating weight matrix similarly
            weight_matrix[current_frame : current_frame + self.window_length] += (
                self.window**2
            )

        # normalizing samples that overlap by dividing by weight matrix
        # only where weight is greater than small amount
        non_zero = weight_matrix > 1e-8
        overlapped_row[non_zero] /= weight_matrix[non_zero]

        return overlapped_row

    # FFT function for frames to the power of two
    def fft(self, signal):
        # getting the length of the signal
        N = len(signal)
        # checking if we're at recursive base
        if N == 1:
            return signal

        else:
            # else divide and conquer
            even = self.fft(signal[::2])
            odd = self.fft(signal[1::2])

            # K represents each frequency bin
            K = np.arange(N)
            base = np.exp(-1j * (2 * np.pi * K / N))

            return np.concatenate(
                [even + base[: int(N / 2)] * odd, even + base[int(N / 2) :] * odd]
            )
