
import os
#-------------------------- set gpu using tf ---------------------------
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
#-------------------  start importing keras module ---------------------
import keras.backend.tensorflow_backend as K
from keras.preprocessing.image import ImageDataGenerator
import argparse
import numpy as np
from Models.SpectrogramModels import *
from AudioProcessing.DataUtils import load_wav_files, AudioSample
from AudioProcessing.TrainUtils import enhance_speech, TensorBoardImage, DataGenerator
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
    #autoencoder = multi_gpu_model(autoencoder, gpus=2)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    autoencoder.compile(optimizer=adam, loss='mean_squared_logarithmic_error')

    print(autoencoder.summary())

    # Reshaping for model
    input_samples = np.reshape(input_samples, (*input_samples.shape, 1))
    target_samples = np.reshape(target_samples, (*target_samples.shape, 1))

    input_datagen = ImageDataGenerator()
    target_datagen = ImageDataGenerator()

    # Provide the same seed and keyword arguments to the fit and flow methods   
    seed = 1
    input_datagen.fit(input_samples, augment=False, seed=seed)
    target_datagen.fit(target_samples, augment=False, seed=seed)

    input_generator = input_datagen.flow(input_samples, None, batch_size=16, seed=seed)

    target_generator = target_datagen.flow(target_samples, None, batch_size=16, seed=seed)
    
    train_generator = zip(input_generator, target_generator)

    autoencoder.fit_generator(generator=train_generator,
                              validation_data=train_generator,
                              steps_per_epoch=input_samples.shape[0]//16,
                              validation_steps=input_samples.shape[0]//16,
                              shuffle=True,
                              epochs=50,
                              verbose=1)

    #autoencoder.evaluate(input_np_data, target_np_data)

    #reconstructed_sample = enhance_speech(autoencoder,)

    #reconstructed_sample.write_wavfile('wav.wav')
