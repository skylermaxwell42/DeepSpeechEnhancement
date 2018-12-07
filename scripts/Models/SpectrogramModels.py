from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, AtrousConvolution1D, Conv2DTranspose, Reshape, ELU
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K

def SpecNet():

    input = Input(shape=(256, 496, 1))

    conv1 = Conv2D(filters=25,
                   kernel_size=(100, 50),
                   strides=2,
                   activation='relu',
                   padding='same')(input)

    conv2 = Conv2DTranspose(filters=25,
                            kernel_size=(100, 50),
                            strides=2,
                            activation='relu',
                            padding='same')(conv1)

    conv3 = Conv2DTranspose(filters=1,
                            kernel_size=(256, 496),
                            strides=1,
                            activation='relu',
                            padding='same')(conv2)

    output = Activation('sigmoid', name='decoder_output')(conv3)

    return input, output