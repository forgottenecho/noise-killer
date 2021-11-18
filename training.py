"""
11/5/2021
Four layers did not work so well, adjusting to 3 to see if  that helps the model generalize.
My thinking is that the fourth compress has too small dimensionality to hold the data.

11/8/2021
Three layers did not go much better. Best model was Loss 20.9 Val_loss 19.0.
My next try will be to create better mel spectrograms, because the current ones seem
to be mostly dark pixels with just a few, scattered semi-bright lines

11/9/2021
Two layers didn't work either. At this point I'm just trying to get the model to overfit,
at least ot prove that the explanatory power exists within the model. may have to do some
more adjusting

11/9/2021
JUST NOTICED THAT THE FREAKING CONV LAYERS HAD LINEAR ACTIVATION! This probably explains why
all the models have had such a hard time

11/10/2021
Enabled GPU for tensorflow, major speedup from 11s -> 2s per epoch

11/11/2021
Adding ReLU activations still produced similar results. I was stepping throuhg the entire runtime
of the program to find other errors and I saw that the convolutional layers have only 2-4 filters.
This is an extreme lack of explanatory power! I am adding U-Net-esque filter scaling, so that the
filters increas as the layers deepen, 64, 128, 256 ...

11/11/2021
Finally the model is fitting to the data! Overfitting, so need to adjust hyper params and maybe add
dropout

11/17/2021
After more training and no good results, I added dropout loyers and eventually just reconstructed
the whole datset. I added a feature to be able to change n_mels, which edits the number of freq
bins each mel spectrogram has.
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
import json

# seed all libraries
def seed_everything(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

# training session class for identifying/reproducing training results
class Session():
    def __init__(self, sample_size, spec_nfft, spec_hop, n_mels, num_layers, starting_filters, lr, split_ratio, debug):
        # dataset construction params
        self.sample_size = sample_size
        self.spec_nfft = spec_nfft
        self.spec_hop = spec_hop
        self.n_mels = n_mels

        # training params
        self.num_layers = num_layers
        self.starting_filters = starting_filters
        self.lr = lr
        self.split_ratio = split_ratio
        self.time_step = None

        # other
        self.debug = debug
        self.time_stamp = None

        self.new_time_stamp()

    # helper function for preprocessing and dataset building
    def _get_training_data(self, sample_size=1024, acceptable_rates=[44100], max_songs=None, spec_nfft=500, spec_hop=50, noise_factor=0.2, n_mels=128):

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
                mel_spec_channel_1 = librosa.feature.melspectrogram(y=song_tensor[:,0], sr=rate, n_fft=spec_nfft, hop_length=spec_hop, n_mels=n_mels)
                mel_spec_channel_2 = librosa.feature.melspectrogram(y=song_tensor[:,1], sr=rate, n_fft=spec_nfft, hop_length=spec_hop, n_mels=n_mels)
                mel_spec_noisy_1 = librosa.feature.melspectrogram(y=song_tensor_noisy[:,0], sr=rate, n_fft=spec_nfft, hop_length=spec_hop, n_mels=n_mels)
                mel_spec_noisy_2 = librosa.feature.melspectrogram(y=song_tensor_noisy[:,1], sr=rate, n_fft=spec_nfft, hop_length=spec_hop, n_mels=n_mels)

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
    def _get_model(self, shape, layers=1, starting_filters=64, dropout_rate=0.25):
        assert layers > 0
        input = keras.layers.Input(shape=shape)
        last_layer = input
        num_filters = starting_filters

        # encoder portion
        for i in range(layers):
            x = keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same', activation='relu')(last_layer)
            x = keras.layers.MaxPool2D()(x)
            x = keras.layers.Dropout(dropout_rate)(x)
            num_filters *= 2
            last_layer = x
        
        num_filters /= 2

        # decoder portion
        for i in range(layers):
            x = keras.layers.Conv2DTranspose(filters=num_filters, kernel_size=3, strides=2, padding='same',  activation='relu')(last_layer)
            x = keras.layers.Dropout(dropout_rate)(x)
            num_filters /= 2
            last_layer = x

        output = keras.layers.Conv2D(shape[-1], 1)(last_layer)

        return keras.Model(input, output)
    
    def _denoise_test(self, examples, model, hop_length, n_fft, save_dir=''):
        # TODO samplerate is currently hardcoded, must fix to be dynamic!
        sample_rate = 44100

        # get denoised specs
        examples_denoised = model.predict(examples)

        # prepare the save location
        save_path = 'models/t{}/audio_test'.format(self.time_stamp)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for i in range(examples.shape[0]):
            example = examples[i]
            example_denoised = examples_denoised[i]

            # spec to audio
            # TODO here we are loosing the stereo by subscripting! fix later
            rebuild = librosa.feature.inverse.mel_to_audio(example[:, :, 0], sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
            rebuild_denoised = librosa.feature.inverse.mel_to_audio(example_denoised[:, :, 0], sr=sample_rate, n_fft=n_fft, hop_length=hop_length)

            # output to file
            # TODO must fix so that output samples == nubmer of input samples. we are losing some samples
            soundfile.write('models/t{}/audio_test/noisy{}.wav'.format(self.time_stamp, i), data=rebuild, samplerate=sample_rate)
            soundfile.write('models/t{}/audio_test/denoised{}.wav'.format(self.time_stamp, i), data=rebuild_denoised, samplerate=sample_rate)
            print('Saving song #{}'.format(i))

    # wrapper function for clean call
    def get_training_data(self):
        # adding possibility for saving/loading datasets to avoid need for reconstruction for every train
        save_path = 'dataset_specs/{}-{}-{}-{}.npy'.format(
            self.sample_size,
            self.spec_nfft,
            self.spec_hop,
            self.n_mels)

        # use a smaller dataset save file for debugging
        if self.debug:
            save_path = save_path.split('.')[0] + '-debug.' + save_path.split('.')[1]

        # check for existing dataset
        if os.path.exists(save_path):
            # dataset has already been made, load it up
            print('\nDataset already exists! Loading from file {}'.format(save_path))
            combined = np.load(save_path)
            dataset = combined[0]
            dataset_noisy = combined[1]
            print('Found {} instances in the dataset.\n'.format(dataset.shape[0]))
        else: 
            # create dataset from scratch

            # only create partial dataset if debugging
            max_songs = None
            if self.debug:
                max_songs = 100

            dataset, dataset_noisy = self._get_training_data(
                sample_size=self.sample_size,
                spec_nfft=self.spec_nfft,
                spec_hop=self.spec_hop,
                n_mels=self.n_mels,
                max_songs=max_songs
            )

            # save dataset for future reuse
            print('Dataset did not exist! Saving to file {}'.format(save_path))
            combined = np.stack([dataset, dataset_noisy])
            np.save(save_path, combined)

        # save internally for when we build the model
        self.instance_shape = dataset[0].shape

        return dataset, dataset_noisy

    # wrapper fucn
    def get_model(self):
        return self._get_model(shape=self.instance_shape, layers=self.num_layers, starting_filters=self.starting_filters)

    # wrapper func
    def denoise_test(self, examples, model):
        self._denoise_test(examples=examples, model=model, hop_length=self.spec_hop, n_fft=self.spec_nfft)
    
    # for file-naming purposes
    def new_time_stamp(self):
        self.time_stamp = str(round(time.time()))

    def load(self, path):
        pass

    def save(self):
        outfile = open('models/t{}/session.json'.format(self.time_stamp),'w')
        json.dump(self.__dict__, outfile, indent=2)
        outfile.close()

# seed the bois
seed_everything(42)

# training session object
# all HYPERPARAMS are to be changed in this constructor
current_session = Session(
    sample_size=4096,
    spec_nfft=256,
    spec_hop=128,
    n_mels=64,

    num_layers=3,
    starting_filters=4,
    lr = 0.001,
    split_ratio = 0.75,

    debug=True,
)


# build the dataset
data, data_noisy = current_session.get_training_data()

# show random spectrogram to see that it works
if current_session.debug:
    show_another = input("Show a random data instance (y/n)? ")

    while show_another == 'y':
        rand = np.random.randint(0, data.shape[0])
        random_spec = data[rand, :, :, 0]
        random_spec_noisy = data_noisy[rand, :, :, 0]
        figs, axs = plt.subplots(2)
        # TODO title the axes for noisy and non-noisy!
        axs[0].imshow(random_spec)
        axs[1].imshow(random_spec_noisy)
        plt.show()

        show_another = input("Show another random data instance (y/n)? ")

# randomize the data
np.random.shuffle(data)
np.random.shuffle(data_noisy)

# split the dataset
# MODEL SHOULD TAKE IN NOISY INPUT AND SPIT OUT DENOISED OUTPUT!
ratio = current_session.split_ratio # TODO i don't like how this breaks my data/process abstractions
crit_index = math.floor(ratio*len(data))
X_train = data_noisy[:crit_index]
X_test = data_noisy[crit_index:]
Y_train = data[:crit_index]
Y_test = data[crit_index:]

# create the model
model = current_session.get_model()
model.summary()

# added forever loop so dataset doesn't have to be completely reconstructed every single time
while True:
    patience = input("How much patience for the epochs? Enter 'exit' to leave: ")

    if patience == 'exit':
        break
    else:
        patience = int(patience)

    # callbacks
    save = keras.callbacks.ModelCheckpoint('models/t'+current_session.time_stamp+'/l{loss:.3g}-vl{val_loss:.3g}', monitor='val_loss', save_best_only=True, save_weights_only=True) # have to str concat bc ModelCheckpoint uses format specifiers already
    es = keras.callbacks.EarlyStopping('val_loss', patience=patience)

    # train the model
    lr = current_session.lr # TODO again with the broken abstraction :/
    epochs = 100000 # will usually end early from the callback
    if current_session.debug:
        epochs = 5

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
    history = model.fit(
        x=X_train, 
        y=Y_train, 
        validation_data=(X_test, Y_test),
        batch_size=32,
        callbacks=[save,es],
        epochs=epochs
    )

    # save current_session to JSON
    current_session.save()

    # save example 3 audio files for audio test of how it sounds
    # TODO pass in BEST model not just last model
    # best_model = 
    current_session.denoise_test(examples=np.vstack([X_train[0:2],X_test[0:2]]), model=model)
    
    # output the training metrics
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    metrics_path = 'models/t'+current_session.time_stamp+'/history.png'
    print('Metrics plot saved to {}'.format(metrics_path))
    plt.savefig(metrics_path)
    plt.close()


print("debug")