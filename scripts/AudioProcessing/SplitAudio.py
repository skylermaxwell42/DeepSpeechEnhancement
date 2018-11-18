'''
    File: SplitAudio.py
    Description: A python function to take in a wav file and split into several 2 second numpy array

'''

from scipy.io import wavfile
import numpy as numpy

# # of splits =  length of np array/(sampling_rate*2)
# given a numpy array, split it up, based on the sampling rate given and return a list of numpy arrays
# each  array will be a 2 second clip

# each 2 second clip will be 40k(with sampling rate of 20k) length, then shift over by 20k and so on so forth
def splitAudio(data, sampling_rate):
	samples = []
	curr = 0

	while(len(data) - curr >= sampling_rate):
		samples.append(data[curr : curr + sampling_rate * 2])
		curr += sampling_rate

	return samples

#test splitAudio function
if __name__ == '__main__':
	wav_path = '/home/maxwels2/Desktop/Nonspeech/n26.wav'
	sample_rate, data  = wavfile.read(wav_path)
	print(len(data))


	split_audio = splitAudio(data, sample_rate)
	print(split_audio[6])
	print(split_audio)