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
    return data[:,:] * ratio

def volume_Increment_Domain(data, startRatio, endRatio):
    """Apply a linear volume increment across the audio data."""
    linspaces = np.array([np.linspace(startRatio, endRatio, len(data[:,0]), True),
                          np.linspace(startRatio, endRatio, len(data[:,0]), True)])
    output = np.multiply(linspaces.T, data)
    return output.astype(np.int16)

def non_Linear_PCM_A_law(data):
    """Apply A-law non-linear PCM transformation."""
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    
    A = 87.6
    output = []
    
    for channels in normalized_data:
        two = []
        for x in channels:
            x_abs = abs(x)
            if x_abs <= (1/A):
                two.append((x_abs * A) / (1 + math.log(A)))
            elif x_abs <= 1:
                two.append((1 + math.log(x_abs * A)) / (1 + math.log(A)))
        output.append(two)
    
    return np.array(output)

def non_Linear_PCM_Mu_law(data):
    """Apply Mu-law non-linear PCM transformation."""
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    
    Mu = 255
    output = []
    
    for pair in normalized_data:
        two = []
        for x in pair:
            x_abs = abs(x)
            two.append((math.log(1 + Mu*x_abs)) / math.log(1 + Mu))
        output.append(two)
    
    return np.array(output)

def delta_modulation():
    
    previous_bit_channel0 = data[0,0]
    previous_bit_channel1 = data[0,1]
    
    delta_channel0 = [0]
    delta_channel1 = [0]
    
    for i in range(1,len(data)):
        
        if data[i,0] > previous_bit_channel0:
            delta_channel0.append(1)
        elif data[i,0] < previous_bit_channel0:
            delta_channel0.append(-1)
        else:
            delta_channel0.append(0)
            
        if data[i,1] > previous_bit_channel1:
            delta_channel1.append(1)
        elif data[i,1] < previous_bit_channel1:
            delta_channel1.append(-1)
        else:
            delta_channel1.append(0)
    
    return np.column_stack((delta_channel0, delta_channel1))

def main():
    # linear_data = data
    # save_wav('linear_pcm.wav', linear_data, sampling_rate)
    # play_audio(linear_data, sampling_rate)
    # print('Playing Linear PCM')
    
    # incremented_data = volume_Increment(data, 2)
    # play_audio(incremented_data, sampling_rate)
    # save_wav('X2.wav', incremented_data, sampling_rate)
    # print("X2.wav")
    
    # incremented_data = volume_Increment(data, 4)
    # play_audio(incremented_data, sampling_rate)
    # save_wav('X4.wav', incremented_data, sampling_rate)
    # print("X4.wav")
    
    # Increment the volume linearly from -2 to 4
    soft_incremented_data = volume_Increment_Domain(data, -2, 4)
    # play_audio(soft_incremented_data, sampling_rate)
    # save_wav('X_Soft(-2to4).wav', soft_incremented_data, sampling_rate)
    
    plt.figure()
    plt.plot(soft_incremented_data[:,0])
    plt.title("Channel 1")

    plt.figure()
    plt.plot(soft_incremented_data[:,1])
    plt.title("Channel 2")

    # Apply A-law transformation and play the audio
    # A_law_data = non_Linear_PCM_A_law(soft_incremented_data)
    # play_audio(A_law_data, sampling_rate)
    
    plt.figure()
    # plt.plot(A_law_data)
    plt.title("A-law Transformation")

    # Apply Mu-law transformation and play the audio
    # Mu_law_data = non_Linear_PCM_Mu_law(soft_incremented_data)
    # play_audio(Mu_law_data, sampling_rate)
    
    plt.figure()
    # plt.plot(Mu_law_data)
    plt.title("Mu-law Transformation")
    
    # Apply Delta modulation and show on plot
    delta = delta_modulation()
    
    plt.figure(figsize=(80,5))
    plt.plot(delta[:,0])
    plt.title("Delta Modulation")
    
    plt.figure(figsize=(80,5))
    plt.plot(delta[:,1])
    plt.title("Delta Modulation")
    
    
if __name__ == "__main__":
    main()
