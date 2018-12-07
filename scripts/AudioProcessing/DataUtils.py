import os
import librosa
import progressbar
import numpy as np
import random as rand
import tensorflow as tf
from scipy.io import wavfile

from .SpectrogramUtils import pretty_spectrogram

def generate_specgram(audio_sample, fft_size, step_size):
    ''' Routine to generate a single spectrogram from an AudioSample

    Parameters:
        audio_sample    (AudioSample)
        fft_size        (int)
        step_size       (int)
    Returns:
        specgram
    '''
    assert(isinstance(audio_sample, AudioSample))
    specgram = pretty_spectrogram(audio_sample.data, log=True, thresh=6, fft_size=512, step_size=64)

    return specgram


def load_wav_files(input_dir):
    ''' Function to load wav files and the corresponding meta data (sampling rate)


    Parameters:
        input_dir:      (str) String specifiying path to directory that contains audio .wav samples
    Return:
        sample_data:    [AudioSample]
    '''
    pb = progressbar.ProgressBar()

    sample_data = []
    file_names = os.listdir(input_dir)
    for rel_path in pb(file_names):
        if (rel_path[0:3] != 'out_'):
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
    denoise_set = np.empty((1, len(input_data), 2))
    denoise_set[1, :, 1] = input_data
    denoise_set[1, :, 1] = target_data
    denoise_set = denoise_set.astype(input_data.dytpe)
    feature = {'label': bytes_feature(tf.compat.as_bytes(label)),
               'denoise_set': bytes_feature(tf.compat.as_bytes(denoise_set.tostring()))}

    return feature
def bytes_feature(value):
    """ Helper Function to return an object that can be fed to the TF Record """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    """ Helper Function to return an object that can be fed to the TF Record """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def data_input_fn(tfrecords_path, batch_size, shuffle=False):
    ''' Function to tell TensorFlow how to parse our custom tfrecord file

    Parameters:
        tfrecords_path      (str)
        batch_size          (int)
        shuffle             (Boolean)

    Returns:
        _input_fn           (Function) Data Input function to be assist with parsing tf record
    '''

    def _parser(record):
        features = {
            'label': tf.FixedLenFeature([], tf.string),
            'denoise_set': tf.FixedLenFeature([], tf.string)
        }

        parsed = tf.parse_single_example(record, features)
        denoise_set = tf.convert_to_tensor(tf.decode_raw(parsed['input'], tf.float32))
        label = parsed['label']

        return denoise_set, label

    def _input_fn():

        if shuffle:
            dataset = (
                tf.data.TFRecordDataset(tfrecords_path)
                    .map(_parser)
                    .batch(batch_size)
                    .shuffle(buffer_size=10000)
                    .repeat(None) # Infinite iterations, lettng the experiment determine #epochs during training
            )
        else:
            dataset = (
                tf.data.TFRecordDataset(tfrecords_path)
                    .map(_parser)
                    .batch(batch_size)
                    .repeat(None)  # Infinite iterations, lettng the experiment determine #epochs during training
            )

        iterator = dataset.make_one_shot_iterator()

        batch_feats, batch_labels = iterator.get_next()

        return batch_feats, batch_labels

    return _input_fn

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
               'Sample Rate:        {}\n'.format(len(self.data), len(self.data)/self.sample_rate, self.sample_rate)

    def pad_sample(self, target_length):
        ''' Class method to randomly extend the our data set with zeros

            *Side effect: self.data is modified
        '''
        length_out = target_length * self.sample_rate
        start_noise = rand.randint(0, (length_out - len(self.data)))
        arrout = np.zeros(int(length_out)).astype(self.data.dtype)

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
        wavfile.write(path, self.sample_rate, self.data.astype(np.int16))

    def serialize_sample(self):
        return