import torch
import torchaudio
from torch.utils.data import Dataset
import glob
import cv2


class MyCustomFSDD(Dataset):
    def __init__(self, data_path, train = True):
        if(train == True):
            self.data_path = glob.glob(data_path + "/yeni-train-image-dataset/*.png")
        else:
            self.data_path = glob.glob(data_path + "/yeni-test-image-dataset/*.png")

    def __getitem__(self, index):

        fn = self.data_path[index]
        img = cv2.imread(fn) 
        digit = int(fn.split("/")[-1].split("_")[0])
        speaker = fn.split("/")[-1].split("_")[1]

        return img, digit, speaker

    def __len__(self):
        return len(self.data_path)
