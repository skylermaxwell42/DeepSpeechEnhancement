import os
import numpy as np
import random as rand
import keras
from keras.layers import Input, Dense, Reshape, Conv1D, MaxPooling1D, Flatten, UpSampling1D
from scipy.io import wavfile

from DataUtils import load_wav_files
from NoiseNet import NoiseNet

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size, input_data, target_data, dim, n_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.input_data = input_data
        self.target_data = target_data
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.input_data[ID]

            # Store target
            y[i,] = self.target_data[ID]

        return X, y


def superCoolDenseNetMagicStuffFuckJake():
	
	input = Input(shape=(800, 1))
	dense2 = Dense(units=800)(input)
	dense3 = Dense(units=400)(dense2)
	dense4 = Dense(units=200)(dense3)
	dense5 = Dense(units=400)(dense4)
	dense6 = Dense(units=800)(dense5)
	dense7 = Dense(units=1)(dense6)
	#reshape = Reshape((1, 32000))(dense5)

	return input, dense7

def poopydick():

	# ENCODER
	input_sig = Input(batch_shape=(None,800,1))
	x = Conv1D(256,32, activation='relu', padding='same')(input_sig)
	x1 = MaxPooling1D(2)(x)
	x2 = Conv1D(256,32, activation='relu', padding='same')(x1)
	x3 = MaxPooling1D(2)(x2)
	x4 = Conv1D(128,16, activation='relu', padding='same')(x3)
	x5 = MaxPooling1D(2)(x4)
	x6 = Conv1D(128,16, activation='relu', padding='same')(x5)
	x7 = MaxPooling1D(2)(x6)
	x8 = Conv1D(64,8, activation='relu', padding='same')(x7)
	x9 = MaxPooling1D(2)(x8)
	flat = Flatten()(x9)
	encoded = Dense(32,activation = 'relu')(flat)
	 
	#print("shape of encoded {}".format(keras.int_shape(encoded)))
	 
	# DECODER 
	x8_ = Conv1D(64, 8, activation='relu', padding='same')(x9)
	x7_ = UpSampling1D(2)(x8_)
	x6_ = Conv1D(128, 16, activation='relu', padding='same')(x7_)
	x5_ = UpSampling1D(2)(x6_)
	x4_ = Conv1D(128, 16, activation='relu', padding='same')(x5_)
	x3_ = UpSampling1D(2)(x4_)
	x2_ = Conv1D(256, 32, activation='relu', padding='same')(x3_)
	x1_ = UpSampling1D(2)(x2_)
	x_ = Conv1D(256, 32, activation='relu', padding='same')(x1_)
	upsamp = UpSampling1D(2)(x_)
	flat = Flatten()(upsamp)
	decoded = Dense(800,activation = 'relu')(flat)
	decoded = Reshape((800,1))(decoded)
	 
	#print("shape of decoded {}".format(keras.int_shape(decoded)))
	 
	return input_sig, decoded




def main():

	batch_size = 32
	n_channels = 1
	shuffle = True

	input_dir = "/home/maxwels2/Documents/elc470/project6/asr/DeepSpeechEnhancement/testing1/noise/"
	target_dir = "/home/maxwels2/Documents/elc470/project6/asr/DeepSpeechEnhancement/testing1/clean/"
	#input_dir = "/home/data/audio/DeepSpeech/wav/aug_test/noise/"
	#target_dir = "/home/data/audio/DeepSpeech/wav/aug_test/clean/"

	input_Aug = load_wav_files(input_dir)
	target_Aug = load_wav_files(target_dir)

	input_data = []
	input_indices = []
	target_data = []
	target_indices = []

	for i, data in enumerate(input_Aug):
		input_data.append(data.data)
		input_indices.append(i)

	for i, data in enumerate(target_Aug):
		target_data.append(data.data)
		target_indices.append(i)

	input_np_data = np.asarray([x for x in input_data])
	target_np_data = np.asarray([x for x in target_data])

	input_np_data = np.resize(input_np_data, (len(input_np_data), 800, 1))
	target_np_data = np.resize(target_np_data, (len(target_np_data), 800, 1))

	dim = (800, )

	training_generator = DataGenerator(list_IDs = target_indices,
									   batch_size = batch_size,
									   input_data = input_np_data,
									   target_data = target_np_data,
									   dim = dim,
									   n_channels = n_channels,
									   shuffle = shuffle)
									   
	validation_generator = DataGenerator(list_IDs = target_indices,
									     batch_size = batch_size,
									     input_data = input_np_data,
									     target_data = target_np_data,
									     dim = dim,
									     n_channels = n_channels,
									     shuffle = shuffle)


	model_input, model_target = poopydick()
	autoencoder = keras.models.Model(model_input, model_target)
	adam = keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	adadelta = keras.optimizers.Adadelta(lr=0.1)
	autoencoder.compile(optimizer = adadelta, loss='mean_absolute_error')
	autoencoder.summary()

	autoencoder.fit_generator(generator = training_generator,
						validation_data = validation_generator,
						steps_per_epoch = training_generator.__len__(),
						validation_steps = validation_generator.__len__(),
						shuffle =True,
						epochs = 500,
						verbose = 1)

if __name__ == "__main__":
	main()