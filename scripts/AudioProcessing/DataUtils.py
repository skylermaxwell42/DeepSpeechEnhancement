import os
import librosa
import numpy as np
import random as rand
import tensorflow as tf
from scipy.io import wavfile

def load_wav_files(input_dir):
    ''' Function to load wav files and the corresponding meta data (sampling rate)


    Parameters:
        input_dir:      (str) String specifiying path to directory that contains audio .wav samples
    Return:
        sample_data:    (dict) Mapping file names to audio samples and sampling rate
                            keys: 'sample_rate', 'data'
    '''
    sample_data = []
    file_names = os.listdir(input_dir)
    for rel_path in file_names:
        full_path = os.path.join(input_dir, rel_path)
        sample_data.append(AudioSample(full_path))

    return sample_data


def write_audio_enhancement_record(output_path, input_samples, target_samples, labels):
    """ Function to Write a set of images and labels to a TF Record

    Parameters:
        output_path:        (str)
        input_samples:      ([AudioSample])
        target_samples:     ([AudioSample])

    Returns:

    """
    writer = tf.python_io.TFRecordWriter(output_path)
    for input_sample, target_sample, label in zip(input_samples, target_samples, labels):
        # Create a feature
        feature = get_audio_sample_feature(label=label,
                                           sample_rate=input_sample.sample_rate,
                                           input_data=input_sample.data,
                                           target_data=target_sample.data)

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    return

def get_audio_sample_feature(label, sample_rate, input_data, target_data):
    ''' Helper Function to get a feature for tensorflow record file writing and reading

    Paramters:
        label           (str)
        sample_rate     (int)
        input_data      (ndarray)
        target_data     (ndarray)

    Returns:
        feature         (dict)
    '''
    feature = {'label': int64_feature(label),
               'sample_rate': int64_feature(sample_rate),
               'input_sample': bytes_feature(tf.compat.as_bytes(input_data.tostring())),
               'target_sample': bytes_feature(tf.compat.as_bytes(target_data.tostring()))}

    return feature
def bytes_feature(self, value):
    """ Helper Function to return an object that can be fed to the TF Record """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(self, value):
    """ Helper Function to return an object that can be fed to the TF Record """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

class AudioSample(object):
    """ Class to represent the data necessary for the augmenation of Audio Files (.wav)

    """
    def __init__(self, path=None, data=None, sample_rate=None):
        if path:
            self.sample_rate, self.data = wavfile.read(path)
        elif data.any()!=None and sample_rate!=None:
            self.data = data
            self.sample_rate = sample_rate
        return

    def __repr__(self):
        return 'AudioSample Object:\n' \
               'Length in Samples:  {}\n' \
               'Length in Seconds:  {}\n' \
               'Sample Rate:        {}'.format(len(self.data), len(self.data)/self.sample_rate, self.sample_rate)

    def pad_sample(self, target_length):
        ''' Class method to randomly extend the our data set with zeros

            *Side effect: self.data is modified
        '''
        length_out = target_length * self.sample_rate
        start_noise = rand.randint(0, (length_out - len(self.data)))
        arrout = np.zeros(length_out).astype(self.data.dtype)

        for i, num in enumerate(self.data):
            arrout[i + start_noise] = num
            i += 1

        self.data = arrout

    def resample(self, target_sample_rate):
        ''' Class method to upsample a sequence to a speficied factor

            *Side effect:   self.data is modified
                            self.sample_data is modified
        '''
        orig_dtype = self.data.dtype
        data = self.data.astype(np.float64)
        self.data = librosa.core.resample(data, self.sample_rate, target_sample_rate).astype(orig_dtype)
        self.sample_rate = target_sample_rate
        return

    def write_wavfile(self, path):
        ''' Funciton to write audio sample to disk

        '''
        wavfile.write(path, self.sample_rate, self.data)

    def serialize_sample(self):
        return