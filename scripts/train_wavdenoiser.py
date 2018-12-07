import argparse
import numpy as np
from Models.NoiseNet import *
from AudioProcessing.DataUtils import load_wav_files, AudioSample
from AudioProcessing.TrainUtils import enhance_speech, TensorBoardAudio
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

    rgroup.add_argument('--input_sample_dir',
                        help='Top level directory to augmented wave samples (must have sub dirs \'clean\' and \'noise\')',
                        required=True,
                        type=str)

    rgroup.add_argument('--target_sample_dir',
                        help='Top level directory to augmented wave samples (must have sub dirs \'clean\' and \'noise\')',
                        required=True,
                        type=str)

    return parser.parse_args()

if __name__ =='__main__':
    args = parse_args()
    input_samples = load_wav_files(args.input_sample_dir)
    target_samples = load_wav_files(args.target_sample_dir)

    sample_rate = input_samples[0].sample_rate
    input_length = len(input_samples[0].data)


    input_tensor, target_tensor = NoiseNetDense()

    autoencoder = Model(input_tensor, target_tensor)
    #autoencoder = multi_gpu_model(autoencoder, gpus=4)
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    autoencoder.compile(optimizer=adam, loss='mean_squared_logarithmic_error')

    input_np_data = np.asarray([x.data for x in input_samples])
    input_np_data = np.resize(input_np_data, (len(input_np_data), 1, 800))
    print(np.max(input_np_data[0]))
    target_np_data = np.asarray([x.data for x in target_samples])
    target_np_data = np.resize(target_np_data, (len(target_np_data), 1, 800))
    print(autoencoder.summary())

    autoencoder.fit(input_np_data, target_np_data,
                    epochs=10,
                    batch_size=8,
                    shuffle=True,
                    validation_data=(input_np_data, target_np_data),
                    callbacks=[TensorBoard(log_dir='./autoencoder',
                                           histogram_freq=1),
                               TensorBoardAudio('Audio Example')])

    autoencoder.evaluate(input_np_data, target_np_data)

    reconstructed_sample = enhance_speech(autoencoder, AudioSample('/home/data/audio/DeepSpeech/wav/aug_test/noise/out_0.wav'))

    reconstructed_sample.write_wavfile('wav.wav')