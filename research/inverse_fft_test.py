import tensorflow_io as tfio
import matplotlib.pyplot as plt
import numpy as np

audio = tfio.audio.AudioIOTensor('dataset/000/000002.mp3')[:1000]
audio = audio[:, 0] # must split into 2 steps because slicing does a type conversion above and fails on double slice
spectrogram = tfio.audio.spectrogram(
    audio, nfft=5000, window=512, stride=1)

print('Audio shape: ', audio.shape)
print('Spec shape: ', spectrogram.shape)
# plt.figure()
# plt.imshow(spectrogram)
# plt.show()


print('debug')