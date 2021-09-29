# import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import os

def get_training_data(sample_size=1024, acceptable_rates=[44100], max_songs=None):
    print('Generating raw audio dataset')

    # will hold raw song samples
    dataset_raw = np.empty((1, sample_size, 2))

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
            if song_tensor.shape[0] < sample_size:
                continue
            if not song_tensor.shape[1] == 2:
                continue
            
            # process and add to dataset
            song_tensor = song_tensor[:sample_size]
            song_tensor = song_tensor.numpy()
            song_tensor = song_tensor.reshape((1, sample_size, 2))
            dataset_raw = np.vstack([dataset_raw, song_tensor])
        
        # reset
        os.chdir('..')
        print('.', end='')

        # don't load whole dataset when testing
        if not max_songs is None and dataset_raw.shape[0] > max_songs:
            break

    # first instance was empty (garbage)
    dataset_raw = dataset_raw[1:]

    print('Raw audio dataset is of size {}'.format(dataset_raw.shape))
    return dataset_raw

data = get_training_data(max_songs=1000)

print("debug")