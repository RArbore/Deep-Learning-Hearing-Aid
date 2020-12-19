from tensorflow.keras import backend as K
from onnx_tf.backend import prepare
from torchvision import transforms
from torch_stft import STFT
import tensorflow as tf
import pytorch2keras
import numpy as np
import soundfile
import random
import torch
import time
import onnx
import math
import stft
import sys
import os

N = 1024
M = 4
nf = 16
stft_obj = stft.STFT(int(math.sqrt(N*M)-1), int(math.sqrt(N*M)/2), int(math.sqrt(N*M)-1), window='boxcar')

cpu = torch.device("cpu")

class DepthwiseConv2d(torch.nn.Module):
    def __init__(self, nin, nout):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = torch.nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class SimpleModel(torch.nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(1, 1)
        )

    def forward(self, input):
        return self.linear(input)

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

model_pytorch = DenoiseNetwork()
model_pytorch.load_state_dict(torch.load("trial22/model.pt"))
model_pytorch = model_pytorch.to(cpu)

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

K.set_image_data_format("channels_first") 

dummy_input = torch.rand(1, 1, 32, 128)
dummy_output = model_pytorch(dummy_input)

model_keras = pytorch2keras.pytorch_to_keras(model_pytorch, dummy_input, [(1, 32, 128,)], verbose=True, change_ordering=True)

K.set_image_data_format("channels_last")

np_input = dummy_input.view(1, 32, 128, 1).detach().numpy()
np_output = dummy_output.view(1, 32, 128, 1).detach().numpy()
tf_input = tf.convert_to_tensor(np_input)
tf_output = tf.convert_to_tensor(np_output)

converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
model_tfl = converter.convert()

with open("trial22/model.tflite", "wb") as f:
    f.write(model_tfl)

# interpreter = tf.lite.Interpreter(model_path="trial22/model.tflite")
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# interpreter.set_tensor(input_details[0]['index'], tf_input)
# interpreter.invoke()

# tfl_output = interpreter.get_tensor(output_details[0]['index'])
