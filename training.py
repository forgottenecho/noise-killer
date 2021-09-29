# import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import os
import librosa

def get_training_data(sample_size=1024, acceptable_rates=[44100], max_songs=None, spec_nfft=500, spec_hop=50):

    # placeholder for later
    dataset = None

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

            # audio file to tensor of samples
            song_tensor = tfio.audio.AudioIOTensor(song)
            rate = song_tensor.rate.numpy()

            # filter out unwanted data
            if not rate in acceptable_rates:
                continue
            if song_tensor.shape[0] < sample_size: # song too small
                continue
            if not song_tensor.shape[1] == 2: # song not stereo
                continue
            
            # slicing and metadata
            sample_rate = song_tensor.rate.numpy()
            song_tensor = song_tensor[:sample_size]
            song_tensor = song_tensor.numpy()

            # generate those lovely spectrograms, one per each audio channel
            mel_spec_channel_1 = librosa.feature.melspectrogram(y=song_tensor[:,0], sr=sample_rate, n_fft=spec_nfft, hop_length=spec_hop)
            mel_spec_channel_2 = librosa.feature.melspectrogram(y=song_tensor[:,1], sr=sample_rate, n_fft=spec_nfft, hop_length=spec_hop)

            # create dataset / append to the dataset
            new_instance = np.empty((1, mel_spec_channel_1.shape[0], mel_spec_channel_2.shape[1], 2))
            new_instance[0, :, :, 0] = mel_spec_channel_1
            new_instance[0, :, :, 1] = mel_spec_channel_2
            if dataset is None:
                dataset = new_instance
            else:
                dataset = np.vstack([dataset, new_instance])
            # print('Dataset shape: {}'.format(dataset.shape))
        
        # reset
        os.chdir('..')
        print('.', end='')

        # don't load whole dataset when testing
        if not max_songs is None and dataset.shape[0] > max_songs:
            break
    
    print('Dataset is of size {}'.format(dataset.shape))
        
    return dataset

data = get_training_data(max_songs=70)

print("debug")