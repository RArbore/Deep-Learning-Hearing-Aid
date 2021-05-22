import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import soundfile
import librosa
import torch
import time
import os

PATH = "new_data/"

speech = []
noise = []

for file_name in os.listdir(PATH):
    if ".wav" in file_name:
        data, rate = librosa.load(PATH+file_name)
        print(len(data) / rate, file_name)
        sound_as_tensor = torch.tensor(data)
        if "Noise" in file_name:
            noise.append(sound_as_tensor)
        else:
            speech.append(sound_as_tensor)

speech = torch.cat(speech)
noise = torch.cat(noise)
torch.save(speech, "NEW_SPEECH.pt")
print(speech.size())
torch.save(noise, "NEW_NOISE.pt")
print(noise.size())
