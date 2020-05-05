import numpy as np
import os
import torch
import torchaudio
import glob
import cv2
from torchaudio import transforms
import skimage
import torch.nn as nn


data_path = glob.glob("/home/mehmet/Desktop/bitirme/codes/fsdd/recordings/wav_files/*.wav") 
save_path = "/home/mehmet/Desktop/bitirme/codes/fsdd/recordings/a" 

#n_fft = 256
n_fft = 64
fbins = n_fft//2 + 1
# If your .wav file is not 8000 Hz you should convert it first with below code line
# spec_transform = nn.Sequential(transforms.Resample(orig_freq = 44100, new_freq = 8000),transforms.Spectrogram(n_fft = n_fft, normalized = True))
spec_transform = transforms.Spectrogram(n_fft = n_fft, normalized = True)
for i in range(len(data_path)): 
    fn = data_path[i]
    digit = int(fn.split("/")[-1].split("_")[0])   # /0_jackson_0.wav
    speaker = fn.split("/")[-1].split("_")[1]
    number = fn.split("/")[-1].split("_")[2].split('.')[0]
    wave, sample_rate = torchaudio.load_wav(fn)
    wave = spec_transform(wave)

    log_spec = (wave + 1e-9).log2()[0, :, :]

    #width = 65
    width = 33
    height = log_spec.shape[0]
    dim = (width, height)
    log_spec = cv2.resize(log_spec.numpy(), dim, interpolation = cv2.INTER_AREA)
    log_spec = torch.from_numpy(log_spec)
    
    skimage.io.imsave(save_path + '/{}_{}_{}.png'.format(str(digit), str(speaker), str(number)), (log_spec))
