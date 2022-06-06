import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib.pyplot as plt

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
            nn.Linear(args.latent_dim, 256),
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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You should experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.conv_blocks = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # return discriminator score for img
        x = self.conv_blocks(img)
        return x


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    criterion = nn.BCELoss()
    d_losses = np.zeros(args.n_epochs)
    g_losses = np.zeros(args.n_epochs)
    real_labels = torch.ones(args.batch_size, 1)
    fake_labels = torch.zeros(args.batch_size, 1)
    epochs=[]
    for epoch in range(args.n_epochs):
        epochs.append(epoch)
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.view(args.batch_size, -1)
            imgs = Variable(imgs)
            # Train Generator
            # ---------------
            optimizer_G.zero_grad()
            noise = torch.randn(args.batch_size, args.latent_dim)

            if torch.cuda.is_available():
                imgs=imgs.to('cuda')
                generator.cuda()
                discriminator.cuda()
                real_labels=real_labels.to('cuda')
                fake_labels=fake_labels.to('cuda')
                noise=noise.to('cuda')

            gen_imgs = generator(noise)
            ifreal = discriminator(gen_imgs)
            g_loss = criterion(ifreal, real_labels)
            g_loss.backward()
            optimizer_G.step()
            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()
            realself=discriminator(imgs)
            ifreal = discriminator(gen_imgs.detach())
            d_loss =(criterion(realself, real_labels)+criterion(ifreal, fake_labels)) /2
            d_loss.backward()
            optimizer_D.step()
            d_loss = d_loss.item()
            g_loss = g_loss.item()
            d_losses[epoch] = d_losses[epoch] * (i / (i + 1.)) + d_loss * (1. / (i + 1.))
            g_losses[epoch] = g_losses[epoch] * (i / (i + 1.)) + g_loss * (1. / (i + 1.))
            print('Epoch: {}, Generator loss: {}, Discriminator loss: {}'.format(epoch,g_losses[epoch],d_losses[epoch]))
            #

            # Save Images
            # -----------
            gen_imgs = gen_imgs.view(gen_imgs.size(0), 1, 28, 28)
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                save_image(gen_imgs[:25],
                           'images/{}.png'.format(batches_done),
                           nrow=5, normalize=True)
        if (epoch+1) % 50 == 0:
            torch.save(generator.state_dict(), './G--{}.pt'.format(epoch+1))
            torch.save(discriminator.state_dict(),  './D--{}.pt'.format(epoch+1))
    fig1 = plt.subplot(2, 1, 1)
    fig2 = plt.subplot(2, 1, 2)
    fig1.plot(epochs, d_losses, c='red', label='discriminator loss')
    fig1.legend()
    fig2.plot(epochs, g_losses, c='green', label='generator loss')
    fig2.legend()
    plt.savefig('./gan.jpg')
    plt.show()



def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, ), (0.5, ))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator()
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    # torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=25,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
