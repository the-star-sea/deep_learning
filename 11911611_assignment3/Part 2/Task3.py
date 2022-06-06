

import os

import torch
import matplotlib.pyplot as plt
from torch import nn
from torchvision.utils import save_image

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You should experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity
        self.conv_blocks = nn.Sequential(
            nn.Linear(100, 256),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()

        )

    def forward(self, z):
        # Generate images from z

        x = self.conv_blocks(z)
        return x



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
model=Generator()

if torch.cuda.is_available():
    model.load_state_dict(torch.load('G--200.pt'))
else:
    model.load_state_dict(torch.load('G--200.pt',map_location='cpu'))
model.eval()
begin = torch.randn(1,100)
end = torch.randn(1,100)

noises = []
num = 9

for i in range(num):
    noise = (num - i) * end + i * begin
    noises.append(noise / num)


for i in range(num):
    if torch.cuda.is_available():
        gen_img = model(noises[i]).cpu()
    else:
        gen_img = model(noises[i])
    gen_img = gen_img.view( gen_img.size(0),1, 28, 28)
    save_image(gen_img[0],
               'interpolation/{}.png'.format(i),
               nrow=5, normalize=True)

