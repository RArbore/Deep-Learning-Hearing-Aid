import random
import torch
import stft
import time
import math
import sys
import os

N = 1024
M = 16
RESIZE_CONSTANTS = [64, 256]
nf = 16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

stft_obj = stft.STFT(int(math.sqrt(N*M)-1), int(math.sqrt(N*M)/2), int(math.sqrt(N*M)-1), window='boxcar').to(device).eval()

current_milli_time = lambda: int(round(time.time() * 1000))

class DepthwiseConv2d(torch.nn.Module):
    def __init__(self, nin, nout):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = torch.nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

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

def run_model(input, model):
    magnitude, phase = stft_obj.transform(input.view(input.size(0), N*M).to(device))
    d_magnitude = model(magnitude.view(-1, 1, RESIZE_CONSTANTS[0], RESIZE_CONSTANTS[1]).to(device))
    return stft_obj.inverse(d_magnitude.view(-1, RESIZE_CONSTANTS[0], int(RESIZE_CONSTANTS[1]/M)).to(device), phase[:, :, int(RESIZE_CONSTANTS[1]*(M-1)/M):].to(device))


rand_input = torch.rand(1001, 1, N*M).to(device)
model = DenoiseNetwork().to(device).eval()
run_model(rand_input[1000:1001], model)
before_time = current_milli_time()
for i in range(1000):
    b_before_time = current_milli_time()

    run_model(rand_input[i:i+1], model)

    b_after_time = current_milli_time()
    #print(b_after_time-b_before_time)

after_time = current_milli_time()
print("Average Inference Time: "+str((after_time-before_time)/1000.0))
