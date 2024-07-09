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

def quantization(data, bit_length):
    """Quantize the audio data to the specified bit length."""
    # Determine the number of levels
    num_levels = 2 ** bit_length
    
    # Normalize data to the range [0, 1]
    data_normalized = (data - data.min()) / (data.max() - data.min())
    
    # Quantize the normalized data
    data_quantized = np.round(data_normalized * (num_levels - 1))
    
    # Scale back to the original range
    data_reconstructed = data_quantized / (num_levels - 1)
    data_reconstructed = data_reconstructed * (data.max() - data.min()) + data.min()
    
    return data_reconstructed.astype(np.int16)


def change_sampling_rate(data, original_sr, new_sr):
    """Change the sampling rate of the audio file."""
    duration = data.shape[0] / original_sr
    new_length = int(duration * new_sr)
    new_data = np.interp(np.linspace(0, len(data), new_length), np.arange(len(data)), data)
    return new_data.astype(np.int16)


    

def main():
    # Changing the sampling rate
    # new_sampling_rate = sampling_rate * 2  # Doubling the speed of the audio
    # new_data = change_sampling_rate(data, sampling_rate, new_sampling_rate)

    # Save and play the audio file with the new sampling rate
    # save_wav('voice1_fast.wav', new_data, new_sampling_rate)
    # play_audio(new_data, new_sampling_rate)
    # print('New Sampling Rate:', new_sampling_rate)

    # Plot the waveform of the new audio file
    # plt.figure()
    # plt.plot(new_data)
    # plt.title('Waveform of the Audio File with New Sampling Rate')
    # plt.show()
    
    # linear_data = data
    # save_wav('linear_pcm.wav', linear_data, sampling_rate)
    # play_audio(linear_data, sampling_rate)
    # print('Playing Linear PCM')
    plt.figure()
    plt.plot(data)
    
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
    play_audio(soft_incremented_data, sampling_rate)
    save_wav('X_Soft(-2to4).wav', soft_incremented_data, sampling_rate)
    
    # plt.figure()
    # plt.plot(soft_incremented_data[:,0])
    # plt.title("Channel 1")

    # plt.figure()
    # plt.plot(soft_incremented_data[:,1])
    # plt.title("Channel 2")

    # Apply A-law transformation and play the audio
    A_law_data = non_Linear_PCM_A_law(soft_incremented_data)
    play_audio(A_law_data, sampling_rate)
    
    # plt.figure()
    # plt.plot(A_law_data)
    # plt.title("A-law Transformation")

    # Apply Mu-law transformation and play the audio
    Mu_law_data = non_Linear_PCM_Mu_law(soft_incremented_data)
    play_audio(Mu_law_data, sampling_rate)
    
    # plt.figure()
    # plt.plot(Mu_law_data)
    # plt.title("Mu-law Transformation")
    
    # Apply Delta modulation and show on plot
    # delta = delta_modulation()
    
    # plt.figure(figsize=(80,5))
    # plt.plot(delta[:,0])
    # plt.title("Delta Modulation")
    
    # plt.figure(figsize=(80,5))
    # plt.plot(delta[:,1])
    # plt.title("Delta Modulation")
    
    # Apply quantization
    quantized_data_2bit = quantization(data, 2)
    quantized_data_4bit = quantization(data, 4)
    quantized_data_8bit = quantization(data, 8)
    
    plt.figure(figsize=(80,20))
    plt.plot(quantized_data_2bit[:,0])
    plt.title("2-bit Quantization Channel 1")
    
    plt.figure(figsize=(80,20))
    plt.plot(quantized_data_2bit[:,1])
    plt.title("2-bit Quantization Channel 2")
    
    plt.figure(figsize=(80,20))
    plt.plot(quantized_data_4bit[:,0])
    plt.title("4-bit Quantization Channel 1")
    
    plt.figure(figsize=(80,20))
    plt.plot(quantized_data_4bit[:,1])
    plt.title("4-bit Quantization Channel 2")
    
    plt.figure(figsize=(80,20))
    plt.plot(quantized_data_8bit[:,0])
    plt.title("8-bit Quantization Channel 1")
    
    plt.figure(figsize=(80,20))
    plt.plot(quantized_data_8bit[:,1])
    plt.title("8-bit Quantization Channel 2")
    
    
if __name__ == "__main__":
    main()
