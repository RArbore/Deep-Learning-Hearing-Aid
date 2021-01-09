import random
import torch
import stft
import time
import math
import sys
import os

N = 1024
M = 4
nf = 16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

stft_obj = stft.STFT(int(math.sqrt(N*M)-1), int(math.sqrt(N*M)/2), int(math.sqrt(N*M)-1), window='boxcar').to(device)

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
        self.s1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, nf, 3, 1, 1),
            torch.nn.ReLU(True),
        )
        self.s2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DepthwiseConv2d(nf, nf * 2),
            torch.nn.ReLU(True),
        )
        self.s3 = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DepthwiseConv2d(nf * 2, nf * 4),
            torch.nn.ReLU(True),
        )
        
        # self.s4 = torch.nn.Sequential(
        #     torch.nn.Conv2d(nf * 4, nf * 2, 3, 1, 1),
        #     torch.nn.LeakyReLU(0.2),
        # )
        self.s5 = torch.nn.Sequential(
            torch.nn.Conv2d(nf, nf, 3, 1, 1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(nf, 1, 1, 1, 0),
        )

        self.upconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(nf * 4, nf * 2, 2, 2)
        )
        self.upconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(nf * 2, nf, 2, 2)
        )

    def forward(self, input):
        s1 = self.s1(input)
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        s4 = self.upconv2(self.upconv1(s3))
        output = self.s5(s4)

        return output

def run_model(input):
    # magnitude, phase = stft_obj.transform(input.view(input.size(0), N*M))
    # d_magnitude = model(magnitude.view(1, 1, 32, 128))
    # return stft_obj.inverse(d_magnitude.view(1, 32, 128), phase)
    return model(input.view(1, 1, 32, 128)).view(1, 1, 4096)

rand_input = torch.rand(1001, 1, 4096).to(device)
model = DenoiseNetwork().to(device)
run_model(rand_input[1000:1001])
before_time = current_milli_time()
for i in range(1000):
    b_before_time = current_milli_time()

    run_model(rand_input[i:i+1])

    b_after_time = current_milli_time()
    print(b_after_time-b_before_time)

after_time = current_milli_time()
print("Average Inference Time: "+str((after_time-before_time)/1000.0))