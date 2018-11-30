import os
import argparse
import numpy as np
import random as rand
from scipy.io import wavfile

from AudioProcessing.DataUtils import load_wav_files
from AudioProcessing.AugTools import split_audio, add_samples


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

    ogroup = parser.add_argument_group('Optional Arguments')

    ogroup.add_argument('--tf_record_path',
                        help='Path to write a TF record to. If excluded only wav files will be written',
                        required=False,
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
    target_sample_rate = 16000

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
    print('Split Noise Audio Samples to produce: {} samples'.format(len(noise_split_samples)))

    #   Most of the Augmentation is happening here
    #   -The functions called on these audio sample objects modify them in place

    # Padding the noise samples
    noise_augmented_samples = []
    for audio_sample in noise_split_samples:
        audio_sample.pad_sample(base_target_length)
        noise_augmented_samples.append(audio_sample)

    # Resampling the noise data
    for audio_sample in noise_augmented_samples:
        audio_sample.resample(target_sample_rate)

    # Resampling the clean data
    clean_augmented_samples = []
    for audio_sample in clean_split_samples:
        audio_sample.resample(target_sample_rate)

        clean_augmented_samples.append(audio_sample)

    #   Combining Noise with the Clean data for use in ML models
    #   -Before this is done the clean and noisy data must be the same length and the same sampling rate
    composite_samples = []
    for i, audio_sample in enumerate(clean_split_samples):
        noise_sample = rand.choice(noise_augmented_samples)
        augmented_sample = add_samples(noise_sample=noise_sample,
                                       audio_sample=audio_sample,
                                       attn_level=0.5)
        augmented_sample.write_wavfile(os.path.join(args.output_dir, 'clean', 'out_{}.wav'.format(i)))
        audio_sample.write_wavfile(os.path.join(args.output_dir, 'noise', 'out_{}.wav'.format(i)))

    print('{}\n'
          'Augmentation Complete:\n'
          'Wrote: {} augmented samples to {}/{}\n'
          'Wrote: {} clean samples to {}/{}'.format('-'*50, i+1, args.output_dir, 'noise', args.output_dir, 'clean'))








