import glob
import torchaudio
from torchaudio import transforms
import matplotlib.pyplot as plt
# List of all wav files in recording directory(2000 wav)
data_path = "/home/fatih/code/free-spoken-digit-dataset/recordings"

wavs = glob.glob(data_path + "/*.wav")
n_fft = 128
spec_transform = transforms.Spectrogram(n_fft=n_fft)

max_spec_size = 0
wav = []
for filename in wavs:
    wave, sr = torchaudio.load_wav(filename)
    wave = spec_transform(wave)
    wav.append(wave.size(2))
    if(wave.size(2) > max_spec_size):
        max_spec_size = wave.size(2)
print(max_spec_size)

plt.figure()
plt.plot(wav)
plt.show()