import numpy as np
import math
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import sounddevice as sd

# Read the audio file
sampling_rate, data = wav.read('../resource/voice1.wav')
print('Sampling rate:', sampling_rate)
print('Data type:', data.dtype)
print('Data shape:', data.shape)

N, no_channels = data.shape
print('Signal length:', N)

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


def volume_increment(data, ratio):
    """Increment the volume of the audio data by a specified ratio."""
    return data[:, :] * ratio


def volume_soft_increment(data, startRatio, endRatio):
    """Apply a linear volume increment across the audio data."""
    linspaces = np.array([np.linspace(startRatio, endRatio, len(data[:, 0]), True),
                          np.linspace(startRatio, endRatio, len(data[:, 0]), True)])
    output = np.multiply(linspaces.T, data)
    return output.astype(np.int16)


def non_linear_pcm_a_law(data):
    """Apply A-law non-linear PCM transformation."""
    normalized_data = (((data - data.min()) / (data.max() - data.min())) * 2) - 1
    A = 87.6
    output = []

    for channels in normalized_data:
        new_pair = []
        for x in channels:
            x_abs = abs(x)
            if x_abs <= (1 / A):
                new_pair.append(np.sign(x) * ((x_abs * A) / (1 + math.log(A))))
            elif x_abs <= 1:
                new_pair.append(np.sign(x) * ((1 + math.log(x_abs * A)) / (1 + math.log(A))))
        output.append(new_pair)

    return np.array(output)


def non_linear_pcm_mu_law(data):
    """Apply Mu-law non-linear PCM transformation."""
    normalized_data = (((data - data.min()) / (data.max() - data.min())) * 2) - 1
    Mu = 255
    output = []

    for pair in normalized_data:
        new_pair = []
        for x in pair:
            x_abs = abs(x)
            new_pair.append(np.sign(x) * math.log(1 + Mu * x_abs) / math.log(1 + Mu))
        output.append(new_pair)

    return np.array(output)


def delta_modulation(data):
    """Apply Delta modulation to the audio data."""
    previous_bit_channel0 = data[0, 0]
    previous_bit_channel1 = data[0, 1]

    delta_channel0 = [0]
    delta_channel1 = [0]

    for i in range(1, len(data)):
        delta_channel0.append(1 if data[i, 0] > previous_bit_channel0 else -1)
        delta_channel1.append(1 if data[i, 1] > previous_bit_channel1 else -1)

    return np.column_stack((delta_channel0, delta_channel1))


def quantization(data, bit_length):
    """Quantize the audio data to the specified bit length."""
    num_levels = 2 ** bit_length
    data_normalized = (data - data.min()) / (data.max() - data.min())
    data_quantized = np.round(data_normalized * (num_levels - 1))
    data_reconstructed = data_quantized / (num_levels - 1)
    data_reconstructed = data_reconstructed * (data.max() - data.min()) + data.min()
    return data_reconstructed.astype(np.int16)


def main():
    # Save and play linear PCM
    linear_data = data
    save_wav('linear_pcm.wav', linear_data, sampling_rate)
    # play_audio(linear_data, sampling_rate)
    plt.figure()
    plt.plot(data[:, 1])
    plt.title("Linear PCM")

    # Volume increment by 2x and 4x
    # incremented_data = volume_increment(data, 2)
    # play_audio(incremented_data, sampling_rate)
    # save_wav('X2.wav', incremented_data, sampling_rate)
    #
    # incremented_data = volume_increment(data, 4)
    # play_audio(incremented_data, sampling_rate)
    # save_wav('X4.wav', incremented_data, sampling_rate)

    # Apply linear volume increment from -2 to 4
    soft_incremented_data = volume_soft_increment(data, -2, 4)
    # play_audio(soft_incremented_data, sampling_rate)
    save_wav('X_Soft(-2to4).wav', soft_incremented_data, sampling_rate)

    # plt.figure()
    # plt.plot(soft_incremented_data[:, 0])
    # plt.title("Channel 1 Soft Increment")
    #
    # plt.figure()
    # plt.plot(soft_incremented_data[:, 1])
    # plt.title("Channel 2 Soft Increment")

    # Apply A-law and Mu-law transformations
    # a_law_data = non_linear_pcm_a_law(soft_incremented_data)
    # play_audio(a_law_data, sampling_rate)
    # plt.figure()
    # plt.plot(a_law_data)
    # plt.title("A-law Transformation")

    # mu_law_data = non_linear_pcm_mu_law(soft_incremented_data)
    # play_audio(mu_law_data, sampling_rate)
    # plt.figure()
    # plt.plot(mu_law_data)
    # plt.title("Mu-law Transformation")

    # Apply Delta modulation
    # delta = delta_modulation(data)
    # plt.figure(figsize=(80, 5))
    # plt.plot(delta[:, 0])
    # plt.title("Delta Modulation Channel 0")
    #
    # plt.figure(figsize=(80, 5))
    # plt.plot(delta[:, 1])
    # plt.title("Delta Modulation Channel 1")
    #
    # # Apply quantization
    quantized_data_2bit = quantization(data, 2)
    quantized_data_4bit = quantization(data, 4)
    quantized_data_8bit = quantization(data, 8)
    #
    plt.figure()
    plt.plot(quantized_data_2bit[:, 0])
    plt.title("2-bit Quantization Channel 1")
    #
    plt.figure()
    plt.plot(quantized_data_2bit[:, 1])
    plt.title("2-bit Quantization Channel 2")
    #
    plt.figure()
    plt.plot(quantized_data_4bit[:, 0])
    plt.title("4-bit Quantization Channel 1")
    #
    plt.figure()
    plt.plot(quantized_data_4bit[:, 1])
    plt.title("4-bit Quantization Channel 2")

    plt.figure()
    plt.plot(quantized_data_8bit[:, 0])
    plt.title("8-bit Quantization Channel 1")

    plt.figure()
    plt.plot(quantized_data_8bit[:, 1])
    plt.title("8-bit Quantization Channel 2")
    #
    # # Speed up X2
    # save_wav('speed_up.wav', data, sampling_rate * 2)
    # play_audio(data, sampling_rate * 2)
    # plt.figure()
    # plt.plot(data[:, 1])
    # plt.title("speed up X2")
    plt.show()


if __name__ == "__main__":
    main()
