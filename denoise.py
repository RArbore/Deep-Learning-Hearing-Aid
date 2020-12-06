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
BATCH_SIZE = 1000
BATCHES_PER_EPOCH = 500
NUM_EPOCHS = 500

N = 2048
M = 8
nf = 8

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

def cartesian_to_polar(input):
    return torch.stack((torch.abs(input), torch.angle(input)), dim=3)

def polar_to_cartesian(mag, phase):
    return torch.view_as_complex(torch.stack((mag[:, :, :]*torch.cos(phase[:, :, :]), mag[:, :, :]*torch.sin(phase[:, :, :])), dim=3))

class DenoiseNetwork(torch.nn.Module):

    def __init__(self):
        super(DenoiseNetwork, self).__init__()
        self.s1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, nf, 3, 1, 1),
            torch.nn.LeakyReLU(0.2),
        )
        self.s2 = torch.nn.Sequential(
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(nf, nf * 2, 3, 1, 1),
            torch.nn.LeakyReLU(0.2),
        )
        self.s3 = torch.nn.Sequential(
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(nf * 2, nf * 4, 3, 1, 1),
            torch.nn.LeakyReLU(0.2),
        )
        
        self.s4 = torch.nn.Sequential(
            torch.nn.Conv2d(nf * 4, nf * 2, 3, 1, 1),
            torch.nn.LeakyReLU(0.2),
        )
        self.s5 = torch.nn.Sequential(
            torch.nn.Conv2d(nf * 2, nf, 3, 1, 1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(nf, 1, 1, 1, 0),
        )

        self.upconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(nf * 4, nf * 2, 2, 2)
        )
        self.upconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(nf * 2, nf, 2, 2)
        )

    def fourier_to_conv(self, input):
        input = input.view(input.size()[0], input.size()[2])
        input = torch.stft(input, int(math.sqrt(N*M)-1), hop_length=int(math.sqrt(N*M)/2), return_complex=True)

        input_polar = cartesian_to_polar(input)
        input_magnitude = input_polar[:, :, :, 0].view(input_polar.size()[0], 1, input_polar.size()[1], input_polar.size()[2])
        input_phase = input_polar[:, :, :, 1]

        s1 = self.s1(input_magnitude)
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        s4 = self.s4(torch.cat((s2, self.upconv1(s3)), dim=1))
        output = self.s5(torch.cat((s1, self.upconv2(s4)), dim=1))

        output = polar_to_cartesian(output.view(input_phase.size()), input_phase)
        output = torch.istft(output, int(math.sqrt(N*M)-1), hop_length=int(math.sqrt(N*M)/2), length=N*M, return_complex=False)
        return output.view(-1, 1, N*M)

    def forward(self, input):
        last_index = len(input.size())-1
        if not input.size()[last_index] % N*M == 0:
            raise Exception("Input not a length multiple of N*M.")
        return self.fourier_to_conv(input)
        

def train_model(speech_data, noise_data):
    model = DenoiseNetwork()

    current_milli_time = lambda: int(round(time.time() * 1000))

    before_time = current_milli_time()
    rand_input = torch.rand(10, 1, N*M)
    for i in range(10):
        model(rand_input[i:i+1, :, :])
    after_time = current_milli_time()
    print("Average Inference Time: "+str((after_time-before_time)/10.0))

    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2))

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
            selection_indices = (torch.rand(BATCH_SIZE, 2) * (DATA_SIZE - N*M)).int()
            for select in range(BATCH_SIZE):
                speech_entry = speech_data[selection_indices[select, 0]:selection_indices[select, 0] + N*M].float()
                speech_batch.append(speech_entry)
                noise_entry = noise_data[selection_indices[select, 1]:selection_indices[select, 1] + N*M].float()
                noise_batch.append(noise_entry)
                w = torch.rand(1)
                noisy_batch.append((w * speech_entry) + ((1 - w) * noise_entry))
            speech_batch = torch.stack(speech_batch).view(BATCH_SIZE, 1, N*M)
            noise_batch = torch.stack(noise_batch).view(BATCH_SIZE, 1, N*M)
            noisy_batch = torch.stack(noisy_batch).view(BATCH_SIZE, 1, N*M)

            output = model(noisy_batch.to(device))
            loss = torch.nn.functional.smooth_l1_loss(output[:, :, N*M-N:], speech_batch[:, :, N*M-N:].to(device))
            loss.backward()
            opt.step()
            epoch_loss += loss.to(cpu).item() / float(BATCHES_PER_EPOCH)

        with torch.no_grad():
            speech_sample_w = speech_data[0:220500].view(1, 1, -1)
            noise_sample_w = noise_data[0:220500].view(1, 1, -1)
            speech_sample = torch.cat((torch.zeros(1, 1, N*(M-1)), speech_sample_w, torch.zeros(1, 1, 25260)), dim=2)
            noise_sample = torch.cat((torch.zeros(1, 1, N*(M-1)), noise_sample_w, torch.zeros(1, 1, 25260)), dim=2)

            w = 0.5
            noisy_sample = (w * speech_sample) + ((1 - w) * noise_sample)
            noisy_sample_w = (w * speech_sample_w) + ((1 - w) * noise_sample_w)

            i = 0
            outputs = []
            while i < speech_sample.size()[2] - N*(M-1):
                app = model(noisy_sample[:, :, i:i+N*M].to(device))[:, :, N*M-N:]
                outputs.append(app)
                i += N
            output = torch.cat(outputs, dim=2)
            output = output[:, :, N*(M-1):220500+N*(M-1)]
            
            soundfile.write(folder + "/epoch"+str(epoch+1) + "/speech_sample.wav", speech_sample_w.view(-1).numpy(), 22050)
            soundfile.write(folder + "/epoch"+str(epoch+1) + "/noise_sample.wav", noise_sample_w.view(-1).numpy(), 22050)
            soundfile.write(folder + "/epoch"+str(epoch+1) + "/noisy_sample.wav", noisy_sample_w.view(-1).numpy(), 22050)
            soundfile.write(folder + "/epoch"+str(epoch+1) + "/output_sample.wav", output.view(-1).to(cpu).numpy(), 22050)

        epoch_after_time = current_milli_time()
        seconds = math.floor((epoch_after_time - epoch_before_time) / 1000)
        minutes = math.floor(seconds / 60)
        seconds = seconds % 60

        print("["+str(epoch + 1)+"]   Loss : "+str(epoch_loss)+"   Took "+str(minutes)+" minute(s) and "+str(seconds)+" second(s).")

        f.write(str(epoch + 1)+" "+str(epoch_loss)+"\n")

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