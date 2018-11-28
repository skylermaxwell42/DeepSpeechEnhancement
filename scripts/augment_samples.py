import os
import argparse
import random as rand
from scipy.io import wavfile

from AudioProcessing.DataUtils import load_wav_files
from AudioProcessing.AugTools import split_audio


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
    noise_audio_file_names = os.listdir(noise_dir)

    clean_samples = load_wav_files(clean_dir)
    noise_samples = load_wav_files(noise_dir)

    print('-'*50)
    print('Sucessfully Loaded in {} clean audio samples'.format(len(clean_samples)))
    print('Sucessfully Loaded in {} noisy audio samples'.format(len(noise_samples)))
    print('{}\nBegining Augmentation'.format('-'*50))

    base_target_length = 2
    noise_target_length = 1

    clean_split_samples = []
    #   Splitting the audio returns a list of audio sample object so we append each
    #   one to an array to keep track of them
    for audio_sample in clean_samples:

        for x in split_audio(audio_sample, base_target_length):
            clean_split_samples.append(x)

    print('Split Clean Audio Samples to produce: {} samples'.format(len(clean_split_samples)))

    noise_split_samples = []
    #   Same procedure as above with a different target length for the split audio clips
    for audio_sample in noise_samples:
        for x in split_audio(audio_sample, noise_target_length):
            noise_split_samples.append(x)

    noise_padded_split_samples = []
    #   Padding the noisy samples to fit the base target length for further augmentation
    for audio_sample in noise_split_samples:
        #   Calling this function on the object modifies the object itself
        audio_sample.pad_sample(base_target_length)
        noise_padded_split_samples.append(audio_sample)

    print('Split and Padded Noise Audio Samples to produce: {} samples'.format(len(noise_padded_split_samples)))






