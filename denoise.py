#https://arxiv.org/pdf/1609.07132.pdf

from torchvision import transforms
import soundfile
import random
import torch
import time
import math
import stft
import sys
import os

manualSeed = int(torch.rand(1).item() * 1000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

VALID_DATA_SIZE = 10000000
DATA_SIZE = 500000000 - VALID_DATA_SIZE
BATCH_SIZE = 200
BATCHES_PER_EPOCH = 500
NUM_EPOCHS = 1000

N = 1024
M = 16
RESIZE_CONSTANTS = [64, 256]
nf = 16

lr = 0.0002
b1 = 0.5
b2 = 0.999

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cpu = torch.device("cpu")

stft_obj = stft.STFT(int(math.sqrt(N*M)-1), int(math.sqrt(N*M)/2), int(math.sqrt(N*M)-1), window='boxcar').to(device)
# stft_obj = stft.STFT(256, 64, 256, window='boxcar').to(device)

folder = ""

class AdaptiveBatchNorm1d(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(AdaptiveBatchNorm1d, self).__init__()
        self.bn = torch.nn.BatchNorm1d(num_features, eps, momentum, affine)
        self.a = torch.nn.Parameter(torch.FloatTensor(1, 1, 1))
        self.b = torch.nn.Parameter(torch.FloatTensor(1, 1, 1))

    def forward(self, x):
        return self.a * x + self.b * self.bn(x)

class DepthwiseConv2d(torch.nn.Module):
    def __init__(self, nin, nout):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = torch.nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

def cartesian_to_polar(input):
    return torch.stack((torch.abs(input), torch.angle(input)), dim=3)

def polar_to_cartesian(mag, phase):
    return torch.view_as_complex(torch.stack((mag[:, :, :]*torch.cos(phase[:, :, :]), mag[:, :, :]*torch.sin(phase[:, :, :])), dim=3))

# def model_processing(input, model):
#     input = input.view(input.size()[0], input.size()[2])
#     input = torch.stft(input, int(math.sqrt(N*M)-1), hop_length=int(math.sqrt(N*M)/2), return_complex=True)
#     input_polar = cartesian_to_polar(input)
#     input_magnitude = input_polar[:, :, :, 0].view(input_polar.size()[0], 1, input_polar.size()[1], input_polar.size()[2])
#     input_phase = input_polar[:, :, :, 1]

#     output = model(input_magnitude)

#     output = polar_to_cartesian(output.view(input_phase.size()), input_phase)
#     output = torch.istft(output, int(math.sqrt(N*M)-1), hop_length=int(math.sqrt(N*M)/2), length=N*M, return_complex=False)
#     return output.view(-1, 1, N*M)

def model_processing(input, model):
    magnitude, phase = stft_obj.transform(input.view(input.size(0), N*M).to(device))
    d_magnitude = model(magnitude.view(-1, 1, RESIZE_CONSTANTS[0], RESIZE_CONSTANTS[1]).to(device))
    return stft_obj.inverse(d_magnitude.view(-1, RESIZE_CONSTANTS[0], int(RESIZE_CONSTANTS[1]/M)).to(device), phase[:, :, int(RESIZE_CONSTANTS[1]*(M-1)/M):].to(device))



class DenoiseNetwork(torch.nn.Module):

    def __init__(self):
        super(DenoiseNetwork, self).__init__()
        # self.s1 = torch.nn.Sequential(
        #     torch.nn.Conv2d(1, nf, 3, 1, 1),
        #     torch.nn.Dropout(0.5),
        #     torch.nn.LeakyReLU(0.2),
        #     torch.nn.Conv2d(nf, nf, 3, 1, 1),
        #     torch.nn.Dropout(0.5),
        #     torch.nn.LeakyReLU(0.2),
        # )
        # self.s2 = torch.nn.Sequential(
        #     torch.nn.MaxPool2d(2),
        #     torch.nn.Conv2d(nf, nf * 2, 3, 1, 1),
        #     torch.nn.Dropout(0.5),
        #     torch.nn.LeakyReLU(0.2),
        #     torch.nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
        #     torch.nn.Dropout(0.5),
        #     torch.nn.LeakyReLU(0.2),
        # )
        # self.s3 = torch.nn.Sequential(
        #     torch.nn.MaxPool2d(2),
        #     torch.nn.Conv2d(nf * 2, nf * 4, 3, 1, 1),
        #     torch.nn.Dropout(0.5),
        #     torch.nn.LeakyReLU(0.2),
        #     torch.nn.Conv2d(nf * 4, nf * 4, 3, 1, 1),
        #     torch.nn.Dropout(0.5),
        #     torch.nn.LeakyReLU(0.2),
        # )
        
        # self.s4 = torch.nn.Sequential(
        #     torch.nn.Conv2d(nf * 4, nf * 2, 3, 1, 1),
        #     torch.nn.Dropout(0.5),
        #     torch.nn.LeakyReLU(0.2),
        #     torch.nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
        #     torch.nn.Dropout(0.5),
        #     torch.nn.LeakyReLU(0.2),
        # )
        # self.s5 = torch.nn.Sequential(
        #     torch.nn.Conv2d(nf * 2, nf, 3, 1, 1),
        #     torch.nn.Dropout(0.5),
        #     torch.nn.LeakyReLU(0.2),
        #     torch.nn.Conv2d(nf, nf, 3, 1, 1),
        #     torch.nn.Dropout(0.5),
        #     torch.nn.LeakyReLU(0.2),
        #     torch.nn.Conv2d(nf, 1, 1, 1, 0),
        # )

        # self.upconv1 = torch.nn.Sequential(
        #     torch.nn.ConvTranspose2d(nf * 4, nf * 2, 2, 2),
        #     torch.nn.Dropout(0.5),
        # )
        # self.upconv2 = torch.nn.Sequential(
        #     torch.nn.ConvTranspose2d(nf * 2, nf, 2, 2),
        #     torch.nn.Dropout(0.5),
        # )

        self.b1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, nf * 2, (9, 241), (1, 1), (4, 0), bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf * 2),
            torch.nn.Conv2d(nf * 2, nf * 4, (5, 1), (1, 1), (2, 0), bias=False),
        )
        self.b2 = torch.nn.Sequential(
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf * 4),
            torch.nn.Conv2d(nf * 4, nf, (9, 1), (1, 1), (4, 0), bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf),
            torch.nn.Conv2d(nf, nf * 2, (9, 1), (1, 1), (4, 0), bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf * 2),
            torch.nn.Conv2d(nf * 2, nf * 4, (5, 1), (1, 1), (2, 0), bias=False),
        )
        self.b3 = torch.nn.Sequential(
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf * 4),
            torch.nn.Conv2d(nf * 4, nf, (9, 1), (1, 1), (4, 0), bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf),
            torch.nn.Conv2d(nf, nf * 2, (9, 1), (1, 1), (4, 0), bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf * 2),
            torch.nn.Conv2d(nf * 2, nf * 4, (5, 1), (1, 1), (2, 0), bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf * 4),
            torch.nn.Conv2d(nf * 4, nf, (9, 1), (1, 1), (4, 0), bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf),
            torch.nn.Conv2d(nf, nf * 2, (9, 1), (1, 1), (4, 0), bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf * 2),
            torch.nn.Conv2d(nf * 2, nf * 4, (5, 1), (1, 1), (2, 0), bias=False),
        )
        self.b4 = torch.nn.Sequential(
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf * 4),
            torch.nn.Conv2d(nf * 4, nf, (9, 1), (1, 1), (4, 0), bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf),
            torch.nn.Conv2d(nf, nf * 2, (9, 1), (1, 1), (4, 0), bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf * 2),
            torch.nn.Conv2d(nf * 2, nf * 4, (5, 1), (1, 1), (2, 0), bias=False),
        )
        self.b5 = torch.nn.Sequential(
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf * 4),
            torch.nn.Conv2d(nf * 4, nf, (9, 1), (1, 1), (4, 0), bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf),
            torch.nn.Dropout2d(0.2, True),
            torch.nn.Conv2d(nf, 1, (1, 1), (1, 1), (0, 0), bias=True),
        )


    def forward(self, input):
        # s1 = self.s1(input)
        # s2 = self.s2(s1)
        # s3 = self.s3(s2)
        # s4 = self.s4(torch.cat((s2, self.upconv1(s3)), dim=1))
        # output = self.s5(torch.cat((s1, self.upconv2(s4)), dim=1))
        # return output

        b1 = self.b1(input)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b4 = self.b4(b3+b2)
        b5 = self.b5(b4+b1)

        return b5
        

def sdr_loss(pred, label):
    return -(torch.sum(label**2)/torch.sum((pred-label)**2))

def train_model(speech_data, noise_data):
    model = DenoiseNetwork().to(device)

    current_milli_time = lambda: int(round(time.time() * 1000))

    # before_time = current_milli_time()
    # rand_input = torch.rand(10, 1, N*M).to(device)
    # for i in range(10):
    #     model_processing(rand_input[i:i+1, :, :], model)
    # after_time = current_milli_time()
    # print("Average Inference Time: "+str((after_time-before_time)/10.0))

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2))

    before_time = current_milli_time()

    print("Beginning Training with N of " + str(N) + ".")
    print("")

    f = open(folder + "/during_training_performance.txt", "a")

    for epoch in range(0, NUM_EPOCHS):
        os.mkdir(folder + "/epoch"+str(epoch+1))

        epoch_loss = 0
        valid_loss = 0
        epoch_before_time = current_milli_time()

        for batch in range(BATCHES_PER_EPOCH):
            opt.zero_grad()
            model = model.train()
            speech_batch = []
            #noise_batch = []
            noisy_batch = []
            selection_indices = (torch.rand(BATCH_SIZE, 2) * (DATA_SIZE - N*M)).int()
            for select in range(BATCH_SIZE):
                speech_entry = speech_data[selection_indices[select, 0]:selection_indices[select, 0] + N*M].float()
                noise_entry = noise_data[selection_indices[select, 1]:selection_indices[select, 1] + N*M].float()
                #noise_batch.append(noise_entry)
                w = torch.rand(1) * 0.7 + 0.2
                speech_batch.append(w * speech_entry)
                noisy_batch.append(((w * speech_entry) + ((1 - w) * noise_entry)))
            speech_batch = torch.stack(speech_batch).view(BATCH_SIZE, 1, N*M)
            #noise_batch = torch.stack(noise_batch).view(BATCH_SIZE, 1, N*M)
            noisy_batch = torch.stack(noisy_batch).view(BATCH_SIZE, 1, N*M)

            output = model_processing(noisy_batch.to(device), model)
            # loss = torch.nn.functional.mse_loss(output[:, :, N*M-N:], speech_batch[:, :, N*M-N:].to(device))
            loss = sdr_loss(output, speech_batch[:, :, N*M-N:].to(device))
            loss.backward()
            opt.step()
            epoch_loss += loss.to(cpu).item() / float(BATCHES_PER_EPOCH)

        with torch.no_grad():
            model = model.eval()
            speech_sample_w = speech_data[DATA_SIZE:DATA_SIZE+VALID_DATA_SIZE].view(1, 1, -1)
            noise_sample_w = noise_data[DATA_SIZE:DATA_SIZE+VALID_DATA_SIZE].view(1, 1, -1)
            speech_sample = torch.cat((torch.zeros(1, 1, N*(M-1)), speech_sample_w, torch.zeros(1, 1, 2432)), dim=2)
            noise_sample = torch.cat((torch.zeros(1, 1, N*(M-1)), noise_sample_w, torch.zeros(1, 1, 2432)), dim=2)

            w = 0.5
            noisy_sample = ((w * speech_sample) + ((1 - w) * noise_sample))
            noisy_sample_w = ((w * speech_sample_w) + ((1 - w) * noise_sample_w))

            valid_iters = 0
            i = 0
            outputs = []
            while i < speech_sample.size()[2] - N*(M-1):
                # print(noisy_sample[:, :, i:i+N*M].to(device).size())
                block_input = noisy_sample[:, :, i:i+N*M].to(device)
                # app = model_processing(block_input, model)[:, :, N*M-N:]
                app = model_processing(block_input, model)
                loss = sdr_loss(app, block_input[:, :, N*M-N:])
                valid_loss += loss.to(cpu).item()
                outputs.append(app)
                i += N
                valid_iters += 1
            output = torch.cat(outputs, dim=2)
            output = output[:, :, N*(M-1):VALID_DATA_SIZE+N*(M-1)]
            
            valid_loss /= valid_iters

            soundfile.write(folder + "/epoch"+str(epoch+1) + "/speech_sample.wav", speech_sample_w.view(-1).numpy(), 22050)
            soundfile.write(folder + "/epoch"+str(epoch+1) + "/noise_sample.wav", noise_sample_w.view(-1).numpy(), 22050)
            soundfile.write(folder + "/epoch"+str(epoch+1) + "/noisy_sample.wav", noisy_sample_w.view(-1).numpy(), 22050)
            soundfile.write(folder + "/epoch"+str(epoch+1) + "/output_sample.wav", output.view(-1).to(cpu).numpy(), 22050)

        epoch_after_time = current_milli_time()
        seconds = math.floor((epoch_after_time - epoch_before_time) / 1000)
        minutes = math.floor(seconds / 60)
        seconds = seconds % 60

        print("["+str(epoch + 1)+"]   Loss : "+str(epoch_loss)+"      Validation Loss : "+str(valid_loss)+"   Took "+str(minutes)+" minute(s) and "+str(seconds)+" second(s).")

        f.write(str(epoch + 1)+" "+str(epoch_loss)+" "+str(valid_loss)+"\n")

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

    if len(sys.argv) <= 1:
        files = os.listdir(".")
        m = [int(f[5:]) for f in files if len(f) > 5 and f[0:5] == "trial"]
        if len(m) > 0:
            folder = "trial" + str(max(m) + 1)
        else:
            folder = "trial1"
    else:
        folder = sys.argv[1]

    os.mkdir(folder)

    print("Created session folder " + folder)

    print("Loading data...")

    speech_data = torch.load("SPEECH.pt").roll(int(torch.rand(1).item() * (DATA_SIZE + VALID_DATA_SIZE)), dims=0)
    noise_data = torch.load("NOISE.pt").roll(int(torch.rand(1).item() * (DATA_SIZE + VALID_DATA_SIZE)), dims=0)
    
    speech_data = speech_data.roll(int(torch.rand(1).item() * speech_data.size(0)), dims=0)
    noise_data = noise_data.roll(int(torch.rand(1).item() * noise_data.size(0)), dims=0)

    after_time = current_milli_time()
    seconds = math.floor((after_time - before_time) / 1000)
    minutes = math.floor(seconds / 60)
    seconds = seconds % 60
    print("Data loading took " + str(minutes) + " minute(s) " + str(seconds) + " second(s).")

    model = train_model(speech_data, noise_data)
