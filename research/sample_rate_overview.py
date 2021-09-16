'''
For the fma small dataset, results are:

Hz, count
{44100: 7575, 48000: 411, 22050: 14}
'''
import tensorflow_io as tfio
import os

# init
stats = {}
os.chdir('dataset')

# loop over whole dataset
for folder in os.listdir():

    # skip files
    if not folder.isnumeric():
        continue

    # get the audio rate
    os.chdir(folder)
    
    # each song in mini-batch
    for song in os.listdir():
        audio = tfio.audio.AudioIOTensor(song)
        rate = audio.rate.numpy()
        print('{} for {}'.format(rate, song))
        
        # log to stats
        if stats.get(rate) is None:
            stats[rate] = 1
        else:
            stats[rate] += 1
    
    # reset
    os.chdir('..')

print(stats)