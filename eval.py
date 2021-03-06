import soundfile
import torch

N = 100
W = 64

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
            torch.nn.Conv1d(1, W, 3, 1, 1, bias=False),
            AdaptiveBatchNorm1d(W),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv1d(W, W, 3, 1, 2**0, 2**0, bias=False),
            AdaptiveBatchNorm1d(W),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv1d(W, W, 3, 1, 2**1, 2**1, bias=False),
            AdaptiveBatchNorm1d(W),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv1d(W, W, 3, 1, 2**2, 2**2, bias=False),
            AdaptiveBatchNorm1d(W),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv1d(W, W, 3, 1, 2**3, 2**3, bias=False),
            AdaptiveBatchNorm1d(W),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv1d(W, W, 3, 1, 2**4, 2**4, bias=False),
            AdaptiveBatchNorm1d(W),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv1d(W, W, 3, 1, 1, 1, bias=False),
            AdaptiveBatchNorm1d(W),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv1d(W, 1, 1, 1, 0, bias=True),
        )

    def forward(self, input):
        return self.conv(input)

with torch.no_grad():
    model = DenoiseNetwork()

    speech_data = torch.load("SPEECH.pt")
    noise_data = torch.load("NOISE.pt")

    speech_sample = speech_data[0:220500].view(1, 1, -1)
    noise_sample = noise_data[0:220500].view(1, 1, -1)
    w = torch.rand(1)
    noisy_sample = (w * speech_sample) + ((1 - w) * noise_sample)

    output = model(noisy_sample)

    print(output)

    print(torch.nn.functional.mse_loss(output, speech_sample))

    soundfile.write("speech_sample.wav", speech_sample.view(-1).numpy(), 22050)
    soundfile.write("noise_sample.wav", noise_sample.view(-1).numpy(), 22050)
    soundfile.write("noisy_sample.wav", noisy_sample.view(-1).numpy(), 22050)
    soundfile.write("output_sample.wav", output.view(-1).numpy(), 22050)