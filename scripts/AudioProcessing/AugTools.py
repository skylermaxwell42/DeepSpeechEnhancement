from scipy.io import wavfile
import numpy as np
import random as rand


def split_audio(data, sampling_rate, target_length):
    # # of splits =  length of np array/(sampling_rate*2)
    # given a numpy array, split it up, based on the sampling rate given and return a list of numpy arrays
    # each  array will be a 2 second clip

    # each 2 second clip will be 40k(with sampling rate of 20k) length, then shift over by 20k and so on so forth
    samples = []
    curr = 0

    while(len(data) - curr >= sampling_rate * target_length):
        samples.append(data[curr : curr + sampling_rate * target_length])
        curr += sampling_rate

    return samples


def pad_noise(noise, sr):
    length_out = 2 * sr
    start_noise = rand.randint(0, (length_out - len(noise)))
    arrout = np.zeros(length_out)

    for i, num in enumerate(noise):
        arrout[i+start_noise] = num
        i += 1
    return arrout

'''
#test splitAudio function
if __name__ == '__main__':
    wav_path = '/home/maxwels2/Desktop/Nonspeech/n27.wav'
    sample_rate, data  = wavfile.read(wav_path)
    print(len(data))


    split_audio = splitAudio(data, sample_rate)
    padded_noise = padNoise(split_audio[0], sample_rate)
    print(len(padded_noise))
'''