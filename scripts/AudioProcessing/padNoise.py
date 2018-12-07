from scipy.io import wavfile
import numpy as np
import random as rand

#Given a numpy array of noise and the sampling rate of clean data,
#return a numpy array of size 2 * sampling rate randomly padded with zeros
def padNoise(noise, sr):
	length_out = 2 * sr
	start_noise = rand.randint(0, (length_out - len(noise)))
	arrout = np.zeros(length_out)

	i = start_noise
	for num in noise:
		arrout[i] = num
		i += 1

	return arrout