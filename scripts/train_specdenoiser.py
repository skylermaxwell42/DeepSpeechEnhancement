import os
import argparse
import numpy as np
from Models.SpectrogramModels import *
from AudioProcessing.DataUtils import load_wav_files, AudioSample
from AudioProcessing.TrainUtils import enhance_speech, TensorBoardImage
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.utils import multi_gpu_model
from keras.optimizers import Adam

def parse_args():
    parser = argparse.ArgumentParser(description='Script to train a Speech Enhancement Neural Network')

    rgroup = parser.add_argument_group('Required Arguments')

    rgroup.add_argument('--model_dir',
                        help='Path to write persistent training data (aka checkpoints and metadata)',
                        required=True,
                        type=str)

    rgroup.add_argument('--input_spec_dir',
                        help='Top level directory to augmented spectrogram samples',
                        required=True,
                        type=str)

    return parser.parse_args()

if __name__ =='__main__':
    args = parse_args()
    input_samples = np.load(os.path.join(args.input_spec_dir, 'noise_specs.npy'))
    target_samples = np.load(os.path.join(args.input_spec_dir, 'clean_specs.npy'))

    sample_rate = 16000
    input_shape = input_samples[0].shape

    print(input_shape)
    print(input_samples[6].shape)
    input_tensor, target_tensor = SpecNet()

    autoencoder = Model(input_tensor, target_tensor)
    #autoencoder = multi_gpu_model(autoencoder, gpus=4)
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    autoencoder.compile(optimizer=adam, loss='mean_squared_logarithmic_error')

    print(autoencoder.summary())

    # Reshaping for model
    input_samples = np.reshape(input_samples, (*input_samples.shape, 1))
    target_samples = np.reshape(target_samples, (*target_samples.shape, 1))

    autoencoder.fit(input_samples, target_samples,
                    epochs=10,
                    batch_size=8,
                    shuffle=True,
                    validation_data=(input_samples, target_samples),
                    callbacks=[TensorBoard(log_dir='./autoencoder',
                                           histogram_freq=1),
                               TensorBoardImage('Audio Example', num_samples=10)])

    autoencoder.evaluate(input_np_data, target_np_data)

    reconstructed_sample = enhance_speech(autoencoder,)

    reconstructed_sample.write_wavfile('wav.wav')