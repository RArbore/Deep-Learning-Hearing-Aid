#Based off of https://arxiv.org/pdf/1806.10522.pdf

from torchvision import transforms
import soundfile
import random
import torch
import time
import math
import sys
import os

manualSeed = int(torch.rand(1).item() * 1000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

DATA_SIZE = 500000000
BATCH_SIZE = 5000
BATCHES_PER_EPOCH = 100
NUM_EPOCHS = 500

N = 100
W = 64

lr = 0.0001
b1 = 0.5
b2 = 0.999

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cpu = torch.device("cpu")

folder = ""

class AdaptiveBatchNorm1d(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(AdaptiveBatchNorm1d, self).__init__()
        self.bn = torch.nn.BatchNorm1d(num_features, eps, momentum, affine)
        self.a = torch.nn.Parameter(torch.FloatTensor(1, 1, 1))
        self.b = torch.nn.Parameter(torch.FloatTensor(1, 1, 1))

    def forward(self, x):
        return self.a * x + self.b * self.bn(x)

class DenoiseNetwork(torch.nn.Module):

    def __init__(self):
        super(DenoiseNetwork, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(1, W, 3, 1, 1, bias=True),
            #torch.nn.BatchNorm1d(W),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv1d(W, W, 3, 1, 2**0, 2**0, bias=True),
            #torch.nn.BatchNorm1d(W),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv1d(W, W, 3, 1, 2**1, 2**1, bias=True),
            #torch.nn.BatchNorm1d(W),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv1d(W, W, 3, 1, 2**2, 2**2, bias=True),
            #torch.nn.BatchNorm1d(W),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv1d(W, W, 3, 1, 2**3, 2**3, bias=True),
            #torch.nn.BatchNorm1d(W),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv1d(W, W, 3, 1, 2**4, 2**4, bias=True),
            #torch.nn.BatchNorm1d(W),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv1d(W, W, 3, 1, 1, 1, bias=True),
            #torch.nn.BatchNorm1d(W),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv1d(W, 1, 1, 1, 0, bias=True),
        )

    def forward(self, input):
        return self.conv(input)

def train_model(speech_data, noise_data):
    model = DenoiseNetwork().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2))

    current_milli_time = lambda: int(round(time.time() * 1000))

    before_time = current_milli_time()

    print("Beginning Training with N of " + str(N) + ".")
    print("")

    f = open(folder + "/during_training_performance.txt", "a")

    for epoch in range(0, NUM_EPOCHS):
        os.mkdir(folder + "/epoch"+str(epoch+1))

        epoch_loss = 0
        epoch_before_time = current_milli_time()

        for batch in range(BATCHES_PER_EPOCH):
            opt.zero_grad()
            speech_batch = []
            noise_batch = []
            noisy_batch = []
            selection_indices = (torch.rand(BATCH_SIZE, 2) * (DATA_SIZE - N)).int()
            for select in range(BATCH_SIZE):
                speech_entry = speech_data[selection_indices[select, 0]:selection_indices[select, 0] + N].float()
                speech_batch.append(speech_entry)
                noise_entry = noise_data[selection_indices[select, 1]:selection_indices[select, 1] + N].float()
                noise_batch.append(noise_entry)
                w = torch.rand(1)
                noisy_batch.append((w * speech_entry) + ((1 - w) * noise_entry))
            speech_batch = torch.stack(speech_batch).view(BATCH_SIZE, 1, N)
            noise_batch = torch.stack(noise_batch).view(BATCH_SIZE, 1, N)
            noisy_batch = torch.stack(noisy_batch).view(BATCH_SIZE, 1, N)

            output = model(noisy_batch.to(device))
            loss = torch.nn.functional.mse_loss(output, speech_batch.to(device))
            loss.backward()
            opt.step()
            epoch_loss += loss.to(cpu).item() / float(BATCHES_PER_EPOCH)

        with torch.no_grad():
            speech_sample = speech_data[0:220500].view(1, 1, -1)
            noise_sample = noise_data[0:220500].view(1, 1, -1)
            w = 0.5
            noisy_sample = (w * speech_sample) + ((1 - w) * noise_sample)

            output = model(noisy_sample.to(device))

            soundfile.write(folder + "/epoch"+str(epoch+1) + "/speech_sample.wav", speech_sample.view(-1).numpy(), 22050)
            soundfile.write(folder + "/epoch"+str(epoch+1) + "/noise_sample.wav", noise_sample.view(-1).numpy(), 22050)
            soundfile.write(folder + "/epoch"+str(epoch+1) + "/noisy_sample.wav", noisy_sample.view(-1).numpy(), 22050)
            soundfile.write(folder + "/epoch"+str(epoch+1) + "/output_sample.wav", output.view(-1).to(cpu).numpy(), 22050)

        epoch_after_time = current_milli_time()
        seconds = math.floor((epoch_after_time - epoch_before_time) / 1000)
        minutes = math.floor(seconds / 60)
        seconds = seconds % 60

        print("["+str(epoch + 1)+"]   Loss : "+str(epoch_loss)+"   Took "+str(minutes)+" minute(s) and "+str(seconds)+" second(s).")

        f.write(str(epoch + 1)+" "+str(epoch_loss))

    after_time = current_milli_time()

    torch.save(model.state_dict(), folder + "/model.pt")
    print("")
    f.close()

    seconds = math.floor((after_time - before_time) / 1000)
    minutes = math.floor(seconds / 60)
    seconds = seconds % 60

    print(str(NUM_EPOCHS) + " epochs took " + str(minutes) + " minute(s) " + str(seconds) + " second(s).")

    return model

if __name__ == "__main__":
    print("Start!")
    current_milli_time = lambda: int(round(time.time() * 1000))
    before_time = current_milli_time()

    files = os.listdir(".")
    m = [int(f[5:]) for f in files if len(f) > 5 and f[0:5] == "trial"]
    if len(m) > 0:
        folder = "trial" + str(max(m) + 1)
    else:
        folder = "trial1"
    os.mkdir(folder)

    print("Created session folder " + folder)

    print("Loading data...")

    speech_data = torch.load("SPEECH.pt")
    noise_data = torch.load("NOISE.pt")

    after_time = current_milli_time()
    seconds = math.floor((after_time - before_time) / 1000)
    minutes = math.floor(seconds / 60)
    seconds = seconds % 60
    print("Data loading took " + str(minutes) + " minute(s) " + str(seconds) + " second(s).")

    model = train_model(speech_data, noise_data)