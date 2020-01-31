import torch
import torchaudio
from torch.utils.data import Dataset
import glob

class MyCustomFSDD(Dataset):
    def __init__(self, data_path, transfrom=None):
        self.data_path = data_path
        self.transform = transfrom

    def __getitem__(self, index):

        fn = self.data_path[index]
        wave, sample_rate = torchaudio.load_wav(fn)
        digit = int(fn.split("/")[-1].split("_")[0])
        speaker = fn.split("/")[-1].split("_")[1]

        if self.transform:
            wave = self.transform(wave)

        return (wave, sample_rate), digit, speaker

    def __len__(self):
        return len(self.data_path)