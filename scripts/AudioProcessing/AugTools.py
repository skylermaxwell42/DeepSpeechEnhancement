from .DataUtils import AudioSample


def add_samples(x, y):
    ''' Fucntion to super impose audio samples (Agumentation method)

    Parameters:
        x   (AudioSample)
        y   (AudioSample)

        *
    Returns:
        x + y
    '''
    return

def split_audio(audio_sample, target_length):
    # # of splits =  length of np array/(sampling_rate*2)
    # given a numpy array, split it up, based on the sampling rate given and return a list of numpy arrays
    # each  array will be a 2 second clip

    # each 2 second clip will be 40k(with sampling rate of 20k) length, then shift over by 20k and so on so forth
    samples = []
    curr = 0

    while(len(audio_sample.data) - curr >= audio_sample.sample_rate * target_length):
        split_data = audio_sample.data[curr : curr + audio_sample.sample_rate * target_length]
        split = AudioSample(data=split_data,
                            sample_rate=audio_sample.sample_rate)
        samples.append(split)
        curr += audio_sample.sample_rate

    return samples