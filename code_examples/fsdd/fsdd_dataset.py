import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import glob
import random
from torch._six import container_abcs


class MyCustomFSDD(Dataset):
    def __init__(self, data_path, train = True, transform = None):
        if(train == True):
            self.data_path = glob.glob(data_path + "/train-dataset/*.wav")
        else:
            self.data_path = glob.glob(data_path + "/test-dataset/*.wav")
        self.transform = transform

    def __getitem__(self, index):

        fn = self.data_path[index]
        wave, sample_rate = torchaudio.load_wav(fn)
        digit = int(fn.split("/")[-1].split("_")[0])
        speaker = fn.split("/")[-1].split("_")[1]

        if self.transform:
            wave = self.transform(wave)

        seq_length = wave.size(2)

        return (wave, sample_rate), digit, speaker, seq_length

    def __len__(self):
        return len(self.data_path)


max_spec_size = 130
# This function is purely specialized for the return structure of MyCustomFSDD class, do not change this.
# Modified version of default_collate() of PyTorch.
def my_collate(batch):

    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        out = None
        lst = list(batch)

        for idx in range(len(batch)): 
            lst[idx] = F.pad(lst[idx], (0, max_spec_size-lst[idx].size(2))) 
        batch = tuple(lst)

        if torch.utils.data.get_worker_info() is not None:
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [my_collate(samples) for samples in transposed]
