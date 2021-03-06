from audio2numpy import open_audio
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import soundfile
import librosa
import torch
import time
import os

LENGTH_TO_CONSTRUCT = 500000000

SPEECH_PATH = "en/clips" #Common Voice dataset, https://commonvoice.mozilla.org/en/datasets
NOISE_PATH = "UrbanSound8k/audio/fold" #UrbanSound8k dataset, https://urbansounddataset.weebly.com/urbansound8k.html

# print(len(os.listdir(SPEECH_PATH)))
# print(len(os.listdir(NOISE_PATH)))

length = 0
sounds_as_tensors = []

for file_name in os.listdir(SPEECH_PATH):
    if ".mp3" in file_name:
        data, rate = open_audio(SPEECH_PATH+"/"+file_name)
        data = librosa.resample(data, 48000, 22050)
        length += len(data)
        sound_as_tensor = torch.tensor(data)
        sounds_as_tensors.append(sound_as_tensor)
        if length >= LENGTH_TO_CONSTRUCT:
            break

speech_tensor = torch.cat(sounds_as_tensors)[:LENGTH_TO_CONSTRUCT]
torch.save(speech_tensor, "SPEECH.pt")
print(speech_tensor.size())

length = 0
sounds_as_tensors = []

for fold in range(1, 11):
    for file_name in os.listdir(NOISE_PATH+str(fold)):
        if ".wav" in file_name:
            data, rate = librosa.load(NOISE_PATH+str(fold)+"/"+file_name)
            length += len(data)
            sound_as_tensor = torch.tensor(data)
            sounds_as_tensors.append(sound_as_tensor)
            if length >= LENGTH_TO_CONSTRUCT:
                break

noise_tensor = torch.cat(sounds_as_tensors)[:LENGTH_TO_CONSTRUCT]
torch.save(noise_tensor, "NOISE.pt")
print(noise_tensor.size())