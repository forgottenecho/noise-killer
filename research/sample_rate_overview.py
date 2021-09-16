import tensorflow_io as tfio
import os

count = 0
stop_count = 100
os.chdir('dataset')

for folder in os.listdir():
    os.chdir(folder)
    first_song = os.listdir()[0]
    audio = tfio.audio.AudioIOTensor(first_song)
    print('{} for {}'.format(audio.rate.numpy(), first_song))
    os.chdir('..')

    if count == stop_count:
        break
    else:
        count += 1

print('debug')