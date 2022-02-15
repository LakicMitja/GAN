import numpy as np
import os
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import models, transforms, datasets

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

bs = 1
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
ngpu = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

data_dir = ".\maps"

data_transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.CenterCrop((256, 512)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset_train = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=data_transform)
dataset_val = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=data_transform)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=bs, shuffle=True, num_workers=0)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=24, shuffle=True, num_workers=0)


def show_image(img, title="Slika", figsize=(5, 5)):
    img = img.numpy().transpose(1, 2, 0)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    img = img * std + mean
    np.clip(img, 0, 1)

    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.title(title)


images,_ = next(iter(dataloader_train))


def weights_init(m):
    name = m.__class__.__name__

    if name.find("Conv") > -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif name.find("BatchNorm") > -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.encoder1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)

        self.encoder2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.encoder3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )

        self.encoder4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )

        self.encoder5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )

        self.encoder6 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )

        self.encoder7 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        )

        self.decoder1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5)
        )

        self.decoder2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512 * 2, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5)
        )

        self.decoder3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512 * 2, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5)
        )

        self.decoder4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512 * 2, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )

        self.decoder5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256 * 2, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.decoder6 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128 * 2, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )

        self.decoder7 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64 * 2, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)

        latent_space = self.encoder7(e6)

        d1 = torch.cat([self.decoder1(latent_space), e6], dim=1)
        d2 = torch.cat([self.decoder2(d1), e5], dim=1)
        d3 = torch.cat([self.decoder3(d2), e4], dim=1)
        d4 = torch.cat([self.decoder4(d3), e3], dim=1)
        d5 = torch.cat([self.decoder5(d4), e2], dim=1)
        d6 = torch.cat([self.decoder6(d5), e1], dim=1)

        out = self.decoder7(d6)

        return out


model_G = Generator(ngpu=1)

if device == "cuda" and ngpu > 1:
    model_G = nn.DataParallel(model_G, list(range(ngpu)))

model_G.apply(weights_init)
model_G.to(device)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.structure = nn.Sequential(
            nn.Conv2d(in_channels=3 * 2, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.structure(x)


model_D = Discriminator(ngpu=1)

if device == "cuda" and ngpu > 1:
    model_D = torch.DataParallel(model_D, list(rang(ngpu)))

model_D.apply(weights_init)
model_D.to(device)

out1 = model_D(torch.cat([images[:,:,:,:256].to(device), images[:,:,:,256:].to(device)], dim=1)).to(device)
out2 = torch.ones(size=out1.shape, dtype=torch.float, device=device)

criterion = nn.BCELoss()

optimizerD = optim.Adam(model_D.parameters(), lr=lr, betas=(beta1, beta2))
optimizerG = optim.Adam(model_G.parameters(), lr=lr, betas=(beta1, beta2))

NUM_EPOCHS=200
model_D.to(device)
model_G.to(device)

L1_lambda = 100

for epoch in range(NUM_EPOCHS + 1):
    print(f"Epoha {epoch + 1}")
    for images, _ in iter(dataloader_train):
        # Učenje diskriminatorja
        # Učenje na pravih podatkih
        model_D.zero_grad()

        # Vnos slikovnih podatkov
        inputs = images[:, :, :, :256].to(device)
        # Pravi ciljni podatki
        targets = images[:, :, :, 256:].to(device)

        real_data = torch.cat([inputs, targets], dim=1).to(device)
        # Označevanje pravih podatkov
        outputs = model_D(real_data)
        labels = torch.ones(size=outputs.shape, dtype=torch.float, device=device)

        lossD_real = 0.5 * criterion(outputs, labels)
        lossD_real.backward()

        # Učenje na lažnih podatkih
        gens = model_G(inputs).detach()

        # Generiranje slikovnih podatkov
        fake_data = torch.cat([inputs, gens], dim=1)
        outputs = model_D(fake_data)
        # Označevanje lažnih podatkov
        labels = torch.zeros(size=outputs.shape, dtype=torch.float, device=device)

        lossD_fake = 0.5 * criterion(outputs, labels)
        lossD_fake.backward()

        optimizerD.step()

        # Učenje generatorja
        for i in range(2):
            model_G.zero_grad()

            gens = model_G(inputs)

            # Združeni generirani podatki
            gen_data = torch.cat([inputs, gens], dim=1)
            outputs = model_D(gen_data)
            labels = torch.ones(size=outputs.shape, dtype=torch.float, device=device)

            lossG = criterion(outputs, labels) + L1_lambda * torch.abs(gens - targets).sum()
            lossG.backward()
            optimizerG.step()

    if epoch % 5 == 0:
        torch.save(model_G, "./generator.pth")
        torch.save(model_D, "./diskriminator.pth")

print("Konec!")

model_G = torch.load("./generator.pth")
test_imgs,_ = next(iter(dataloader_val))

satellite = test_imgs[:,:,:,:256].to(device)
maps = test_imgs[:,:,:,256:].to(device)

gen = model_G(satellite)

satellite = satellite.detach().cpu()
gen = gen.detach().cpu()
maps = maps.detach().cpu()

show_image(torchvision.utils.make_grid(satellite, padding=10), title="Satelitski posnetek", figsize=(50,50))
show_image(torchvision.utils.make_grid(gen, padding=10), title="Generirana slika", figsize=(50,50))
show_image(torchvision.utils.make_grid(maps, padding=10), title="Pričakovana slika", figsize=(50,50))
