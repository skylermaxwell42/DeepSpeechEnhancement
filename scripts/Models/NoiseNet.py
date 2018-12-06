from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, AtrousConvolution1D, Conv2DTranspose, Reshape
import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

def NoiseNet():
    ''' NoiseNet
    '''

    input = Input(shape=(32000, 1))  # (batch, depth, sample_length)

    ''' Conv1:
        ------

        input_shape:    (1, 1, 32000)
        filter_size:    (1, 16000)
        stride:         2
        padding:        same

        output_shape:   (128, 1, 32000)    
    '''
    conv1 = Conv1D(filters=25, kernel_size=8000, strides=1, activation='relu', padding='same')(input)
    conv2 = Conv1D(filters=25, kernel_size=6000, strides=1, activation='relu', padding='same')(conv1)

    # at this point the representation is (64, 1, 16000)
    #max_pool1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv2)

    reshape = Reshape((1, 8000, 1, 1))(conv2)
    conv3 = Conv1D(filters=25, kernel_size=8000, strides=1, activation='relu', padding='same')(reshape)

    decoded = Conv1D(filters=1, kernel_size=4000, strides=1, padding='same')(conv3)

    return input, decoded

def NoiseNetV2():
    # Network parameters
    input_shape = (32000, 1, 1)

    kernel_size = 3
    latent_dim = 16
    # Encoder/Decoder number of CNN layers and filters per layer
    layer_filters = [32, 64]

    # Build the Autoencoder Model
    # First build the Encoder Model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    # Stack of Conv2D blocks
    # Notes:
    # 1) Use Batch Normalization before ReLU on deep networks
    # 2) Use MaxPooling2D as alternative to strides>1
    # - faster but not as good as strides>1
    for filters in layer_filters:
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=2,
                   activation='relu',
                   padding='same')(x)

    # Shape info needed to build Decoder Model
    shape = K.int_shape(x)

    # Generate the latent vector
    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_vector')(x)

    # Instantiate Encoder Model
    encoder = Model(inputs, latent, name='encoder')
    encoder.summary()

    # Build the Decoder Model
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    # Stack of Transposed Conv2D blocks
    # Notes:
    # 1) Use Batch Normalization before ReLU on deep networks
    # 2) Use UpSampling2D as alternative to strides>1
    # - faster but not as good as strides>1
    for filters in layer_filters[::-1]:
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=2,
                            activation='relu',
                            padding='same')(x)

    x = Conv2DTranspose(filters=1,
                        kernel_size=kernel_size,
                        padding='same')(x)

    outputs = Activation('sigmoid', name='decoder_output')(x)