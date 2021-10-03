import tensorflow_io as tfio
import tensorflow as tf
import soundfile

# params
noise_factor = 0.2

# load up the audio
audio = tfio.audio.AudioIOTensor('dataset/000/000002.mp3')

# slicing and metadata
sample_rate = audio.rate.numpy()
audio = audio[:200000]
audio = audio[:, 0].numpy()

# add the noise
noisy = audio + noise_factor * abs(audio).mean() * tf.random.normal(audio.shape)

# save files to hear comparison
soundfile.write('research/noisy_audio_testing/orig.wav', data=audio, samplerate=sample_rate)
soundfile.write('research/noisy_audio_testing/noisy.wav', data=noisy, samplerate=sample_rate)