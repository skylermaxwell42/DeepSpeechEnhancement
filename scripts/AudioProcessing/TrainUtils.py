import keras
import random as r
import numpy as np
import tensorflow as tf
from .AugTools import split_audio
from .DataUtils import AudioSample, load_wav_files
import matplotlib.pyplot as plt

def enhance_speech(autoencoder, audio_sample):
    ''' Routine for enhancing speech via an Autoencoder Deep Neural Network

    Arguments:

    Returns:
    '''

    sample_rate = 8000
    model_input_size = 800
    assert(isinstance(audio_sample, AudioSample))

    audio_sample.resample(sample_rate)
    audio_sample_split = [x.data for x in split_audio(audio_sample, 0.1)]

    audio_sample_split = np.reshape(audio_sample_split, (len(audio_sample_split), 1, model_input_size))

    decoded = autoencoder.predict(audio_sample_split)

    output = np.empty((1, 0))
    for x in range(0, 20):
        output = np.append(output, decoded[x])

    return AudioSample(data=output * (2 ** 16), sample_rate=8000)

def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)

class TensorBoardImage(keras.callbacks.Callback):
    def __init__(self, tag, num_samples):
        super().__init__()
        self.tag = tag
        self.sample_indicies = []
        self.num_samples = num_samples

    def on_epoch_end(self, epoch, logs={}):
        if self.sample_indicies == []:
            self.sample_indicies = r.sample(range(0, len(self.model.validation_data)), self.num_samples)

        for x in self.sample_indicies:
            wav_spectrogram = self.model.validation_data[x]
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 4))
            cax = ax.matshow(np.transpose(wav_spectrogram), interpolation='nearest', cmap=plt.cm.afmhot, origin='lower')
            fig.colorbar(cax)
            '''
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            # Convert PNG buffer to TF image
            image = tf.image.decode_png(buf.getvalue(), channels=4)

            # Add the batch dimension
            image = tf.expand_dims(image, 0)
            '''
            image = make_image(fig)
            summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
            writer = tf.summary.FileWriter('./logs')
            writer.add_summary(summary, epoch)
        writer.close()

        return

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