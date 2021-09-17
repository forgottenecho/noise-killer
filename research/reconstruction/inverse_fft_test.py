"""
https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0

basically a combination of regular spectrogram but on mel scale
supposedly it is better for retaining the original audio singal

Reconstruction thoughts:
- after some experimentation it seems plausible to get a decent "sounding"
    reconstruction even if the actual phase is off
- 
"""
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import librosa
import soundfile

# load up the audio
audio = tfio.audio.AudioIOTensor('dataset/000/000002.mp3')

# slicing and metadata
series = audio[:200000]
series = series[:, 0] # must split into 2 steps because slicing does a type conversion above and fails on double slice
series = series.numpy()
sample_rate = audio.rate.numpy()

# encode and decode
n_fft = 500
hop_length = 50
mel_spec = librosa.feature.melspectrogram(y=series, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
rebuild = librosa.feature.inverse.mel_to_audio(mel_spec, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
print('Spectro size: {}'.format(mel_spec.shape))

# this may not be a good measure of error because I believe
# that phase information will be lost and contribute greatly
# to the error but not necessarily change the "sound" to a person
avg_loss = ((rebuild - series) / series).mean()
print('Average reconstruction loss was {}'.format(avg_loss))

# output the audio
soundfile.write('research/reconstruction/orig.wav', data=series, samplerate=sample_rate)
soundfile.write('research/reconstruction/rebuild.wav', data=rebuild, samplerate=sample_rate)

print('debug')