import os
import argparse
import progressbar
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

from AudioProcessing.DataUtils import load_wav_files
from AudiProcessing.TrainUtils import enhance_speech


def parse_args():
    parser = argparse.ArgumentParser(description='Script to train a Speech Enhancement Neural Network')

    rgroup = parser.add_argument_group('Required Arguments')

    rgroup.add_argument('--input_wav_dir',
                        help='Top level directory to augmented spectrogram samples',
                        required=True,
                        type=str)

    rgroup.add_argument('--model_path',
                        help='Path to model.h5',
                        required=True,
                        type=str)

    rgroup.add_argument('--num_samples',
                        help='Number of images to process',
                        required=True,
                        type=int)

    rgroup.add_argument('--output_dir',
                        help='Directory to output spectrogram images to',
                        required=True,
                        type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    input_samples = load_wav_files(args.input_wav_dir)

    autoencoder = load_model(args.model_dir)

    for i, audio_sample in enumerate(input_samples):
        reconstructed = enhance_speech(autoencoder, audio_sample)
        audio_sample.write_wavfile(os.path.join(args.output_dir, 'input_sample_{}.wav'.format(i)))
        reconstructed.write_wavfile(os.path.join(args.output_dir, 'reconstructed_{}.wav'.format(i)))

