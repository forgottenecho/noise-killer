import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import math

np.random.seed(42)

def get_training_data(sample_size=1024, acceptable_rates=[44100], max_songs=None, spec_nfft=500, spec_hop=50, noise_factor=0.2):

    # placeholder for later
    dataset = None
    dataset_noisy = None

    print('Generating dataset')

    # loop through dataset
    os.chdir('dataset')
    for folder in os.listdir():
        
        # skip files
        if not folder.isnumeric():
            continue
            
        # loop through each song
        os.chdir(folder)
        for song in os.listdir():

            # audio file to tensor of samples / metadata
            song_tensor = tfio.audio.AudioIOTensor(song)
            rate = song_tensor.rate.numpy()
            shape = song_tensor.shape.numpy()

            # filter out unwanted data
            if not rate in acceptable_rates:
                continue
            if shape[0] < sample_size: # song too small
                continue
            if not shape[1] == 2: # song not stereo
                continue
            
            # slicing, pick random window of size 'sample_size'
            start_point = np.random.randint(0, shape[0] - sample_size + 1)
            try:
                song_tensor = song_tensor[start_point:start_point+sample_size]
            except tf.errors.InvalidArgumentError: # some songs don't like being sliced towards the END of the array
                print('Did not like the slicing of {}'.format(song))
                continue

            # numpy-ify and create noisy audio sample
            # using max() was too much noise, settled on abs().mean()
            song_tensor = song_tensor.numpy()
            song_tensor_noisy = song_tensor + noise_factor * abs(song_tensor).mean() * tf.random.normal(song_tensor.shape)
            song_tensor_noisy = song_tensor_noisy.numpy()

            # generate those lovely spectrograms, one per each audio channel
            mel_spec_channel_1 = librosa.feature.melspectrogram(y=song_tensor[:,0], sr=rate, n_fft=spec_nfft, hop_length=spec_hop)
            mel_spec_channel_2 = librosa.feature.melspectrogram(y=song_tensor[:,1], sr=rate, n_fft=spec_nfft, hop_length=spec_hop)
            mel_spec_noisy_1 = librosa.feature.melspectrogram(y=song_tensor_noisy[:,0], sr=rate, n_fft=spec_nfft, hop_length=spec_hop)
            mel_spec_noisy_2 = librosa.feature.melspectrogram(y=song_tensor_noisy[:,1], sr=rate, n_fft=spec_nfft, hop_length=spec_hop)

            # create dataset / append to the dataset
            new_instance = np.empty((1, mel_spec_channel_1.shape[0], mel_spec_channel_2.shape[1], 2))
            new_instance[0, :, :, 0] = mel_spec_channel_1
            new_instance[0, :, :, 1] = mel_spec_channel_2
            
            new_instance_noisy = np.empty((1, mel_spec_noisy_1.shape[0], mel_spec_noisy_2.shape[1], 2))
            new_instance_noisy[0, :, :, 0] = mel_spec_noisy_1
            new_instance_noisy[0, :, :, 1] = mel_spec_noisy_2

            if dataset is None:
                dataset = new_instance
                dataset_noisy = new_instance_noisy
            else:
                dataset = np.vstack([dataset, new_instance])
                dataset_noisy = np.vstack([dataset_noisy, new_instance_noisy])
            # print('Dataset shape: {}'.format(dataset.shape))
        
        # reset
        os.chdir('..')
        print('.', end='')

        # don't load whole dataset when testing
        if not max_songs is None and dataset.shape[0] > max_songs:
            break
    
    print('Dataset is of size {}'.format(dataset.shape))

    return dataset, dataset_noisy

# build the dataset
data, data_noisy = get_training_data(max_songs=50, sample_size=4096, spec_nfft=512, spec_hop=64)

# show random spectrogram to see that it works
rand = np.random.randint(0, data.shape[0])
random_spec = data[rand, :, :, 0]
random_spec_noisy = data_noisy[rand, :, :, 0]
figs, axs = plt.subplots(2)
axs[0].imshow(random_spec)
axs[1].imshow(random_spec_noisy)
plt.show()

# randomize the data
np.random.shuffle(data)
np.random.shuffle(data_noisy)

# split the dataset
ratio = 0.75
crit_index = math.floor(ratio*len(data))
X_train = data[:crit_index]
X_test = data[crit_index:]
Y_train = data_noisy[:crit_index]
Y_test = data_noisy[crit_index:]

print("debug")