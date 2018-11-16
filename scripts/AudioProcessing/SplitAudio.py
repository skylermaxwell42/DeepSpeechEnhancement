'''
    File: SplitAudio.py
    Description: A python function to take in a wav file and split into several 2 second numpy array

'''

from scipy.io import wavfile
import numpy as numpy

# # of splits =  length of np array/(sampling_rate*2)
# given a numpy array, split it up, based on the sampling rate given and return a list of numpy arrays
# each  array will be a 2 second clip

# each 2 second clip will be 40k length, then shift over by 20k and so on so forth
def splitAudio(wav, sampling_rate):

	Ignore
