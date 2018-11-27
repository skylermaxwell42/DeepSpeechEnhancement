import os
from scipy.io import wavfile


def augment_samples(samples, ):
    ''' Function to set a pipeline up for audio augmentation

    Parameters:

    Returns:
    '''
    return



def load_wav_files(input_dir):
    ''' Function to load wav files and the corresponding meta data (sampling rate)


    Parameters:
        input_dir:      (str) String specifiying path to directory that contains audio .wav samples
    Return:
        sample_data:    (dict) Mapping file names to audio samples and sampling rate
                            keys: 'sample_rate', 'data'
    '''
    sample_data = {}
    file_names = os.listdir(input_dir)
    for rel_path in file_names:
        full_path = os.path.join(input_dir, rel_path)
        sample_rate, data = wavfile.read(full_path)
        sample_data[rel_path] = {'sample_rate': sample_rate,
                                   'data': data}

    return sample_data


