import os
import argparse
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

from AudioProcessing.DataUtils import load_wav_files, generate_specgram, AudioSample
from AudiProcessing.TrainUtils import enhance_speech
from AudioProcessing.SpectrogramUtils import invert_pretty_spectrogram

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
        spcgram = np.transpose(generate_specgram(audio_sample))

        reconstructed_specgram = autoencoder.predict(np.reshape(1, 256, 496, 1))

        reconstructed_sample = invert_pretty_spectrogram(reconstructed_specgram)

        AudioSample(reconstructed_sample).write_wav_file(args.output_dir+'/enhanced_wav_{}.wav'.format(i))

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 4))
        cax = ax.matshow(spcgram, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot,
                         origin='lower')
        fig.colorbar(cax)
        plt.title('Target Spctrogram')
        fig.savefig(os.path.join(args.output_dir, 'clean_spectrogram_{}.png'.format(i)))
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 4))
        cax = ax.matshow(reconstructed_specgram, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot,
                         origin='lower')
        fig.colorbar(cax)
        plt.title('Noisy Spectrogram')
        fig.savefig(os.path.join(args.output_dir, 'noise_spectrogram_{}.png'.format(i)))
        i = i + 1

    '''
    for i, audio_sample in enumerate(input_samples):
        reconstructed = enhance_speech(autoencoder, audio_sample)
        audio_sample.write_wavfile(os.path.join(args.output_dir, 'input_sample_{}.wav'.format(i)))
        reconstructed.write_wavfile(os.path.join(args.output_dir, 'reconstructed_{}.wav'.format(i)))
    '''
