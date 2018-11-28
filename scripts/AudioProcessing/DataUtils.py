import os
import numpy as np
import random as rand
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
        arrout = np.zeros(length_out)

        for i, num in enumerate(self.data):
            arrout[i + start_noise] = num
            i += 1

        self.data = arrout

    def upsample(self, factor):
        ''' Class method to upsample a sequence to a speficied factor

            *Side effect: self.data is modified
        '''
        return