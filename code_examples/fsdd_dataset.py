from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import torch.nn as nn
import numpy as np
import glob
from scipy.io import wavfile
import models


# Change this
folder_data = glob.glob("C:\\Users\\fatih\\grad\\free-spoken-digit-dataset\\recordings\\*.wav")

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


class MyCustomFSDD(Dataset):
    def __init__(self, data_path, transfrom=None):
        self.data_path = data_path
        self.transform = transfrom

    def __getitem__(self, index):
        fn = self.data_path[index]
        sample_rate, wave = wavfile.read(fn)
        #wave = torch.from_numpy(wave)
        digit = int(fn.split("\\")[-1].split("_")[0])
        speaker = fn.split("\\")[-1].split("_")[1]

        if self.transform:
            wave = self.transform(wave)

        return (wave, sample_rate), digit, speaker

    def __len__(self):
        return len(self.data_path)




