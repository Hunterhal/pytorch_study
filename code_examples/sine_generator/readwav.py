from scipy.io import wavfile
import numpy as np
import torch
import matplotlib.pyplot as plt

filename = "zero.wav"

sample_rate, audio = wavfile.read(filename)  # Returns numpy array.

audio_torch = torch.tensor(audio)            # Convert numpy array to torch tensor.  

# Plot wav file.
plt.figure()
plt.plot(audio)
plt.grid()
plt.show()
