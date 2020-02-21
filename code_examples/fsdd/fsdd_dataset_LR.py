import torch
import librosa
from torch.utils.data import Dataset
import glob
import random
import numpy as np


class MyCustomFSDD(Dataset):
    def __init__(self, data_path, train = True):
        if(train == True):
            self.data_path = glob.glob(data_path + "/train-dataset/*.wav")
        else:
            self.data_path = glob.glob(data_path + "/test-dataset/*.wav")
        #self.transform = transform

    def __getitem__(self, index):

        fn = self.data_path[index]
        wave, sample_rate = librosa.load(fn)
        n_FFT = 256
        wave_stft = librosa.stft(wave, n_fft = n_FFT, dtype = np.float64, win_length = n_FFT, hop_length = n_FFT // 2)
        digit = int(fn.split("\\")[-1].split("_")[0])   # DOĞRU ÇALIŞIYOR!
        speaker = fn.split("/")[-1].split("_")[1]       # DOĞRU ÇALIŞIYOR!
        wave_stft = torch.from_numpy(wave_stft)

        min_wave = 5
        wave_len = wave_stft.size(1)     # wave_stft.size() = [fbins, frames in time] veriyor
        if min_wave == wave_len:
            random_index = 0
        else:
            random_index = random.randint(0, wave_len - min_wave)
        wave_stft = wave_stft[:, random_index : random_index + min_wave]
        return (wave_stft, sample_rate), digit, speaker

    def __len__(self):
        return len(self.data_path)

