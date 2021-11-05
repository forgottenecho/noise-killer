"""
11/5/2021
Four layers did not work so well, adjusting to 3 to see if  that helps the model generalize.
My thinking is that the fourth compress has too small dimensionality to hold the data.
"""
import tensorflow as tf
import tensorflow.keras as keras # parameter hints are broken unless I do this
import tensorflow_io as tfio
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import librosa
import soundfile
import time

# seed all libraries
def seed_everything(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

# helper function for preprocessing and dataset building
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
            print('Folder: {}\tSong: {}\tTotal: {}'.format(folder, song, dataset.shape[0]), end='\r')
        
        # reset
        os.chdir('..')

        # don't load whole dataset when testing
        if not max_songs is None and dataset.shape[0] > max_songs:
            break
    
    # should currently be in 'dataset', go back to project root so 'os' works properly later on
    os.chdir('..')
    
    print('Dataset is of size {}'.format(dataset.shape))

    return dataset, dataset_noisy

# helper function for model architecture
def get_model(shape, layers=1):
    assert layers > 0
    input = keras.layers.Input(shape=shape)
    last_layer = input

    # encoder portion
    for i in range(layers):
        x = keras.layers.Conv2D(filters=4, kernel_size=3, padding='same')(last_layer)
        x = keras.layers.MaxPool2D()(x)
        last_layer = x
    
    # decoder portion
    for i in range(layers):
        x = keras.layers.Conv2DTranspose(filters=2, kernel_size=3, strides=2, padding='same')(last_layer)
        last_layer = x

    output = last_layer

    return keras.Model(input, output)

def denoise_test(examples, model, hop_length, n_fft, save_dir=''):
    # TODO samplerate is currently hardcoded, must fix to be dynamic!
    sample_rate = 44100

    # get denoised specs
    examples_denoised = model.predict(examples)

    for i in range(examples.shape[0]):
        example = examples[i]
        example_denoised = examples_denoised[i]

        # spec to audio
        # TODO here we are loosing the stereo by subscripting! fix later
        rebuild = librosa.feature.inverse.mel_to_audio(example[:, :, 0], sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
        rebuild_denoised = librosa.feature.inverse.mel_to_audio(example_denoised[:, :, 0], sr=sample_rate, n_fft=n_fft, hop_length=hop_length)

        # output to file
        # TODO orig is a bad name because it is actually noised, not the original file!
        # TODO must fix so that output samples == nubmer of input samples. we are losing some samples
        soundfile.write('orig{}.wav'.format(i), data=rebuild, samplerate=sample_rate)
        soundfile.write('rebuild{}.wav'.format(i), data=rebuild_denoised, samplerate=sample_rate)
        print('Saving song #{}'.format(i))

# seed the bois
seed_everything(42)

# build the dataset
data, data_noisy = get_training_data(max_songs=100000, sample_size=4096, spec_nfft=511, spec_hop=64)

# # show random spectrogram to see that it works
# rand = np.random.randint(0, data.shape[0])
# random_spec = data[rand, :, :, 0]
# random_spec_noisy = data_noisy[rand, :, :, 0]
# figs, axs = plt.subplots(2)
# axs[0].imshow(random_spec)
# axs[1].imshow(random_spec_noisy)
# plt.show()

# randomize the data
np.random.shuffle(data)
np.random.shuffle(data_noisy)

# split the dataset
# MODEL SHOULD TAKE IN NOISY INPUT AND SPIT OUT DENOISED OUTPUT!
ratio = 0.75
crit_index = math.floor(ratio*len(data))
X_train = data_noisy[:crit_index]
X_test = data_noisy[crit_index:]
Y_train = data[:crit_index]
Y_test = data[crit_index:]

# create the model
model = get_model(shape=data[0].shape, layers=3)
model.summary()

# added forever loop so dataset doesn't have to be completely reconstructed every single time
while True:
    patience = int(input("How much patience for the epochs? "))
    # callbacks
    time_stamp = str(round(time.time()))
    save = keras.callbacks.ModelCheckpoint('models/l{loss:.3g}-vl{val_loss:.3g}-t'+time_stamp, monitor='val_loss', save_best_only=True) # have to str concat bc ModelCheckpoint uses format specifiers already
    es = keras.callbacks.EarlyStopping('val_loss', patience=patience)

    # train the model
    model.compile(optimizer='Adam', loss='mse')
    history = model.fit(
        x=X_train, 
        y=Y_train, 
        validation_data=(X_test, Y_test),
        batch_size=32,
        callbacks=[save,es],
        epochs=100000
    )

    # save example 3 audio files for audio test of how it sounds
    # TODO have some object that holds dataset construction params! so i don't have to keep passing into functions separately
    denoise_test(examples=np.vstack([X_train[0:2],X_test[0:2]]), model=model, hop_length=64, n_fft=511)
    
    # output the training metrics
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()


print("debug")