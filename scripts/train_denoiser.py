import argparse
import numpy as np
from Models.NoiseNet import NoiseNet, NoiseNetV2
from AudioProcessing.DataUtils import load_wav_files
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.utils import multi_gpu_model

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




    batch_size = 128
    # Instantiate Decoder Model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # Autoencoder = Encoder + Decoder
    # Instantiate Autoencoder Model
    autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    autoencoder.summary()

    autoencoder.compile(loss='mse', optimizer='adam')

    # Train the autoencoder
    autoencoder.fit(x_train_noisy,
                    x_train,
                    validation_data=(x_test_noisy, x_test),
                    epochs=30,
                    batch_size=batch_size)





'''
    input_tensor, target_tensor = NoiseNet()

    autoencoder = Model(input_tensor, target_tensor)
    autoencoder = multi_gpu_model(autoencoder, gpus=4)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    input_np_data = np.asarray([x.data for x in input_samples])
    input_np_data = np.resize(input_np_data, (len(input_np_data), 32000, 1))
    target_np_data = np.asarray([x.data for x in target_samples])
    target_np_data = np.resize(target_np_data, (len(target_np_data), 32000, 1))
    print(autoencoder.summary())

    autoencoder.fit(input_np_data, target_np_data,
                    epochs=50,
                    batch_size=8,
                    shuffle=True,
                    callbacks=[TensorBoard(log_dir='./autoencoder')])
'''