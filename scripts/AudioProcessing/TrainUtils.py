import keras
import random as r
import numpy as np
import tensorflow as tf
from .AugTools import split_audio
from .DataUtils import AudioSample, load_wav_files

def enhance_speech(autoencoder, audio_sample):
    ''' Routine for enhancing speech via an Autoencoder Deep Neural Network

    Arguments:

    Returns:
    '''

    sample_rate = 8000
    model_input_size = 800
    assert(isinstance(audio_sample, AudioSample))

    audio_sample.resample(sample_rate)
    audio_sample_split = [x.data for x in split_audio(audio_sample, 0.1)]

    audio_sample_split = np.reshape(audio_sample_split, (len(audio_sample_split), 1, model_input_size))

    decoded = autoencoder.predict(audio_sample_split)

    output = np.empty((1, 0))
    for x in range(0, 20):
        output = np.append(output, decoded[x])

    return AudioSample(data=output * (2 ** 16), sample_rate=8000)
