"""
Really helpful videos
FFT: https://www.youtube.com/watch?v=z7X6jgFnB6Y
STFFT (spectro): https://www.youtube.com/watch?v=-Yxj3yfvY-4

What I have gathered from videos & experimentation:
nfft = frame_size
window = window_size
stride = hop_size

num_freq_bins = nfft/2 + 1
    - note how the dependency is on frame_size (nfft) NOT window_size!
    - half of freq's are lost because they are the symmetrical "negative" frequencies
    - the plus one I guess is for f=0 (DC signal)
    - normally DFT alg uses whole sample_size so I was expecting num_freq_bins = len(sample)/2 + 1
        however, STFFT takes in a frame and passes THAT to DFT

window vs frame
    - frame can be longer than window, and frame is what is actually passed in
    - window is the sliding window across your sample, where stride is the movement of the window
    - so according to the second video, num_output_steps = ( len(samples) - frame_size ) / stride + 1,
        however, this is only true if frame_size == window_size which is USUALLY the case, but in reality,
        the correct formula is num_output_steps = ( len(samples) - window_size ) / stride + 1
"""
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import numpy as np

audio = tfio.audio.AudioIOTensor('dataset/000/000002.mp3')[:2000]
audio = audio[:, 0] # must split into 2 steps because slicing does a type conversion above and fails on double slice

a = tfio.audio.spectrogram(
    audio, nfft=100, window=10, stride=10)
print(a.shape)

b = tfio.audio.spectrogram(
    audio, nfft=200, window=512, stride=1)
print(b.shape)

c = tfio.audio.spectrogram(
    audio, nfft=300, window=512, stride=1)
print(c.shape)

fig, axs = plt.subplots(3)
axs[0].imshow(a)
axs[1].imshow(b)
axs[2].imshow(c)
plt.show()


print('debug')