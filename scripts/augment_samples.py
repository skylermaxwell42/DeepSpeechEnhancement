from scipy.io import wavfile
import os
import argparse

from AudioProcessing.DataUtils import load_wav_files
from AudioProcessing.AugTools import split_audio, pad_noise

def parse_args():
    parser = argparse.ArgumentParser(description='Script to augment audio samples with noise')

    rgroup = parser.add_argument_group('Required Arguments')

    rgroup.add_argument('--input_dir',
                        help='Input directory of clean audio samples',
                        required=True,
                        type=str)

    rgroup.add_argument('--noise_dir',
                        help='Input directory of noisy audio samples to add to clean samples',
                        required=True,
                        type=str)

    rgroup.add_argument('--output_dir',
                        help='Directory to ouput augmented audio samples to',
                        required=True,
                        type=str)

    return parser.parse_args()

if __name__ == '__main__':
    args =  parse_args()

    clean_dir = args.input_dir
    noise_dir = args.noise_dir
    output_dir = args.output_dir

    clean_audio_file_names = os.listdir(clean_dir)
    noisy_audio_file_names = os.listdir(noise_dir)

    clean_samples = load_wav_files(clean_dir)
    noise_samples = load_wav_files(noise_dir)
    print('Sucessfully Loaded in {} audio samples'.format(len(clean_samples.items())))

    padded_noise = {}
    noise_clip_length = 1                                  #seconds
    output_length = 2
    for id in noise_samples.keys():
        sample = noise_samples[id]
        if len(sample['data'])/sample['sample_rate'] > noise_clip_length:
            splits = split_audio(sample['data'], sample['sample_rate'], noise_clip_length)
            for i, split in enumerate(splits):
                split = pad_noise(split, sample['sample_rate'], output_length)
                padded_noise['{}_{}'.format(id, i)] = {'sample_rate': sample['sample_rate'],
                                                       'data': split}
        else:
            padded_sample = pad_noise(sample['data'], sample['sample_rate'], output_length)
            padded_noise['{}'.format(id)] = {'sample_rate': sample['sample_rate'],
                                                   'data': padded_sample}

    #   Verifying split and padded samples
    for key in padded_noise.keys():
        sample_rate = padded_noise[key]['sample_rate']
        size = len(padded_noise[key]['data'])
        time_length = size/sample_rate
        assert(output_length == time_length) # Length of split and padded samples must be == to target length to cont.






