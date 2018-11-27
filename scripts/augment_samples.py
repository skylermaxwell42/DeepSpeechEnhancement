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
    target_length = 2                                   #seconds
    for id in noise_samples.keys():
        sample = noise_samples[id]
        if len(sample['data'])/sample['sample_rate'] > target_length:
            splits = split_audio(sample['data'], sample['sample_rate'], target_length)
            for split in splits:
                print('Too long')





