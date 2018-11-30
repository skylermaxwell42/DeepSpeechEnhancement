from .DataUtils import AudioSample


def add_samples(noise_sample, audio_sample, attn_level):
    ''' Fucntion to super impose audio samples (Agumentation method)

    Parameters:
        nosie_sample    (AudioSample)
        audio_sample    (AudioSample)
        attn_level      (double) [0.0 < x <= 1.0]

        *
    Returns:
        new_sample      (AudioSample) Composite signall
    '''

    assert(isinstance(noise_sample, AudioSample))
    assert(isinstance(audio_sample, AudioSample))
    assert(noise_sample.sample_rate == audio_sample.sample_rate)
    assert(len(noise_sample.data) == len(audio_sample.data))

    attenuated_noise =  (noise_sample.data*attn_level).astype(noise_sample.data.dtype)
    new_sample = AudioSample(data=(attenuated_noise + audio_sample.data),
                             sample_rate=audio_sample.sample_rate)
    return new_sample

def split_audio(audio_sample, target_length):
    # # of splits =  length of np array/(sampling_rate*2)
    # given a numpy array, split it up, based on the sampling rate given and return a list of numpy arrays
    # each  array will be a 2 second clip

    # each 2 second clip will be 40k(with sampling rate of 20k) length, then shift over by 20k and so on so forth
    assert(isinstance(audio_sample, AudioSample))

    samples = []
    curr = 0

    while(len(audio_sample.data) - curr >= audio_sample.sample_rate * target_length):
        split_data = audio_sample.data[curr : curr + audio_sample.sample_rate * target_length]
        split = AudioSample(data=split_data,
                            sample_rate=audio_sample.sample_rate)
        samples.append(split)
        curr += audio_sample.sample_rate

    return samples