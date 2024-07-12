import numpy as np
import math
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import sounddevice as sd
from sklearn.preprocessing import MinMaxScaler

# Read the audio file
sampling_rate, data = wav.read('../resource/voice1.wav')
print('sampling rate:', sampling_rate)
print('data type:', data.dtype)
print('data shape:', data.shape)

N, no_channels = data.shape
print('signal length:', N)

# Separate the channels
channel0 = data[:, 0]
channel1 = data[:, 1]


def save_wav(filename, data, samplerate):
    """Save the modified data to a wav file."""
    wav.write(filename, samplerate, data)


def play_audio(data, samplerate):
    """Play the audio using sounddevice."""
    sd.play(data, samplerate)
    sd.wait()


def volume_Increment(data, ratio):
    """Increment the volume of the audio data by a specified ratio."""
    return data[:, :] * ratio


def volume_soft_increment(data, startRatio, endRatio):
    """Apply a linear volume increment across the audio data."""
    linspaces = np.array([np.linspace(startRatio, endRatio, len(data[:, 0]), True),
                          np.linspace(startRatio, endRatio, len(data[:, 0]), True)])
    output = np.multiply(linspaces.T, data)
    return output.astype(np.int16)

def expansion(data, original_sr, new_sr):
    """Change the sampling rate of the audio file."""
    duration = data.shape[0] / original_sr
    new_length = int(duration * new_sr)
    
    new_channel0 = np.zeros((new_length))
    new_channel1 = np.zeros((new_length))
    
    i = 0
    for c0, c1 in zip(channel0, channel1):
        new_channel0[i] = c0
        new_channel1[i] = c1
        if i >= 2:
            new_channel0[i-1] = (new_channel0[i-2] + c0) / 2
            new_channel1[i-1] = (new_channel1[i-2] + c1) / 2
        i += 2
        
        
    return np.column_stack((new_channel0, new_channel1)).astype(np.int16)



def main():

    linear_data = data
    save_wav('linear_pcm.wav', linear_data, sampling_rate)
    play_audio(linear_data, sampling_rate * 2)
    print('Playing Linear PCM')
    plt.figure()
    plt.plot(data[:,1])

if __name__ == "__main__":
    main()
    plt.show()
