from matplotlib import pyplot as plt
import torch
import os
from PIL import Image

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader

import numpy as np
from itertools import chain
import torch.nn as nn
from torch.nn import init
import torch.backends.cudnn as cudnn


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_lr_scheduler(optimizer):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.75)
    return scheduler


def set_requires_grad(networks, requires_grad=False):
    for network in networks:
        for param in network.parameters():
            param.requires_grad = requires_grad


def denorm(x):
    out = (x+1) / 2
    return out.clamp(0, 1)


def sample_images(satellite_loader, map_loader, generator_1, generator_2, epoch, path):
    satellite = next(iter(satellite_loader))
    maps = next(iter(map_loader))

    real_A = satellite.to(device)
    real_B = maps.to(device)

    fake_B = generator_1(real_A)
    fake_A = generator_2(real_B)

    fake_ABA = generator_2(fake_B)
    fake_BAB = generator_1(fake_A)

    images = [real_A, fake_B, fake_ABA, real_B, fake_A, fake_BAB]

    result = torch.cat(images, dim=0)
    save_image(denorm(result.data),
               os.path.join(path, 'CycleGAN_maps_%03d.png' % (epoch + 1)),
               nrow=6,
               normalize=True)

    del images


class MapDataset(Dataset):
    def __init__(self, image_path, sort):
        self.path = os.path.join(image_path, sort)
        self.images = [x for x in sorted(os.listdir(self.path))]

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        image_path = os.path.join(self.path, self.images[index])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)


def get_maps_loader(purpose, batch_size):
    if purpose == 'train':
        train_satellite = MapDataset('./data/maps/', 'trainA')
        train_maps = MapDataset('./data/maps/', 'trainB')

        trainloader_satellite = DataLoader(train_satellite, batch_size=batch_size, shuffle=True)
        trainloader_maps = DataLoader(train_maps, batch_size=batch_size, shuffle=True)

        return trainloader_satellite, trainloader_maps

    elif purpose == 'test':
        test_satellite = MapDataset('./data/maps/', 'testA')
        test_maps = MapDataset('./data/maps/', 'testB')

        testloader_satellite = DataLoader(test_satellite, batch_size=batch_size, shuffle=True)
        testloader_maps = DataLoader(test_maps, batch_size=batch_size, shuffle=True)

        return testloader_satellite, testloader_maps


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.in_channel = 3
        self.ndf = 64
        self.out_channel = 1

        model = [
            nn.Conv2d(self.in_channel, self.ndf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        n_blocks = 3

        for i in range(n_blocks):
            mult = 2 ** i
            model += [
                nn.Conv2d(self.ndf * mult, self.ndf * mult * 2, kernel_size=4, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(self.ndf * mult * 2),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        model += [nn.Conv2d(self.ndf * mult * 2, self.out_channel, kernel_size=4, stride=1, padding=1, bias=True),
                  nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, n_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(n_features, n_features, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(n_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(n_features, n_features, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(n_features)
        ]
        self.conv_block = nn.Sequential(*conv_block)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x + self.conv_block(x)
        out = self.relu(out)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.in_channel = 3
        self.ngf = 64
        self.out_channel = 3
        self.num_residual_blocks = 9

        # Initial Block
        model = [
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(self.in_channel, self.ngf, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(self.ngf),
            nn.ReLU(inplace=True)
        ]

        n_downsampling = 3

        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(self.ngf * mult, self.ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(self.ngf * mult * 2),
                nn.ReLU(inplace=True)
            ]

        # Res Blocks
        mult = 2 ** n_downsampling

        for i in range(self.num_residual_blocks):
            model += [ResidualBlock(self.ngf * mult)]

        n_upsampling = 3

        for i in range(n_upsampling):
            mult = 2 ** (n_upsampling - i)
            model += [
                nn.ConvTranspose2d(self.ngf * mult, int(self.ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                   output_padding=1, bias=True),
                nn.InstanceNorm2d(int(self.ngf * mult / 2)),
                nn.ReLU(inplace=True)
            ]

        model += [
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(self.ngf, self.out_channel, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out


# Ponovljivost
cudnn.deterministic = True
cudnn.benchmark = False

# Konfiguracija - CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())

num_epoch = 10


def train():

    torch.manual_seed(5)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(5)

    # Poti za primere, uteži in rezultate
    paths = ['./results/samples/', './results/weights/', './results/plots/']
    paths = [make_dirs(path) for path in paths]

    # Priprava Dataloaderja
    train_satellite_loader, train_maps_loader = get_maps_loader(purpose='train', batch_size=1)
    test_satellite_loader, test_maps_loader = get_maps_loader(purpose='test', batch_size=1)
    total_batch = min(len(train_satellite_loader), len(train_maps_loader))

    # Priprava omrežij
    D_A = Discriminator()
    D_B = Discriminator()
    G_A2B = Generator()
    G_B2A = Generator()

    networks = [D_A, D_B, G_A2B, G_B2A]

    for network in networks:
        network.to(device)

    # Funkcije izgube
    criterion_Adversarial = nn.MSELoss()
    criterion_Cycle = nn.L1Loss()
    criterion_Identity = nn.L1Loss()

    # Optimizacija
    D_A_optim = torch.optim.Adam(D_A.parameters(), lr=2e-4, betas=(0.5, 0.999))
    D_B_optim = torch.optim.Adam(D_B.parameters(), lr=2e-4, betas=(0.5, 0.999))
    G_optim = torch.optim.Adam(chain(G_A2B.parameters(), G_B2A.parameters()), lr=2e-4, betas=(0.5, 0.999))

    D_A_optim_scheduler = get_lr_scheduler(D_A_optim)
    D_B_optim_scheduler = get_lr_scheduler(D_B_optim)
    G_optim_scheduler = get_lr_scheduler(G_optim)

    # Izgube
    D_losses_A, D_losses_B, G_losses = [], [], []

    # Učenje
    print("Št epoh: {}.".format(num_epoch))
    for epoch in range(num_epoch):
        for i, (satellite, maps) in enumerate(zip(train_satellite_loader, train_maps_loader)):

            # Priprava podatkov
            real_A = satellite.to(device)
            real_B = maps.to(device)

            # Inicializacija za optimizacijo
            G_optim.zero_grad()
            D_A_optim.zero_grad()
            D_B_optim.zero_grad()

            # Učenje generatorja
            set_requires_grad([D_A, D_B], requires_grad=False)

            # Nasprotovalna izguba
            fake_A = G_B2A(real_B)
            prob_fake_A = D_A(fake_A)
            real_labels = torch.ones(prob_fake_A.size()).to(device)
            G_mse_loss_B2A = criterion_Adversarial(prob_fake_A, real_labels)

            fake_B = G_A2B(real_A)
            prob_fake_B = D_B(fake_B)
            real_labels = torch.ones(prob_fake_B.size()).to(device)
            G_mse_loss_A2B = criterion_Adversarial(prob_fake_B, real_labels)

            # Izguba identitete
            identity_A = G_B2A(real_A)
            G_identity_loss_A = 5 * criterion_Identity(identity_A, real_A)

            identity_B = G_A2B(real_B)
            G_identity_loss_B = 5 * criterion_Identity(identity_B, real_B)

            # Izguba cikla
            reconstructed_A = G_B2A(fake_B)
            G_cycle_loss_ABA = 10 * criterion_Cycle(reconstructed_A, real_A)

            reconstructed_B = G_A2B(fake_A)
            G_cycle_loss_BAB = 10 * criterion_Cycle(reconstructed_B, real_B)

            # Kalkulacija skupne izgube generatorja
            G_loss = G_mse_loss_B2A + G_mse_loss_A2B + G_identity_loss_A + G_identity_loss_B + G_cycle_loss_ABA + G_cycle_loss_BAB

            # Vzvratno razširjanje in posodobitev
            G_loss.backward()
            G_optim.step()

            # Učenje diskriminatorja
            set_requires_grad([D_A, D_B], requires_grad=True)

            # Učenje diskriminatorja A
            # Prava izguba
            prob_real_A = D_A(real_A)
            real_labels = torch.ones(prob_real_A.size()).to(device)
            D_real_loss_A = criterion_Adversarial(prob_real_A, real_labels)

            # Lažna izguba
            fake_A = G_B2A(real_B)
            prob_fake_A = D_A(fake_A.detach())
            fake_labels = torch.zeros(prob_fake_A.size()).to(device)
            D_fake_loss_A = criterion_Adversarial(prob_fake_A, fake_labels)

            # Kalkulacija skupne izgube diskriminatorja A
            D_loss_A = 5 * (D_real_loss_A + D_fake_loss_A).mean()

            # Vzvratno razširjanje in posodobitev
            D_loss_A.backward()
            D_A_optim.step()

            # Učenje diskriminatorja B
            # Prava izguba
            prob_real_B = D_B(real_B)
            real_labels = torch.ones(prob_real_B.size()).to(device)
            loss_real_B = criterion_Adversarial(prob_real_B, real_labels)

            # Lažna izguba
            fake_B = G_A2B(real_A)
            prob_fake_B = D_B(fake_B.detach())
            fake_labels = torch.zeros(prob_fake_B.size()).to(device)
            loss_fake_B = criterion_Adversarial(prob_fake_B, fake_labels)

            # Kalkulacija skupne izgube diskriminatorja B
            D_loss_B = 5 * (loss_real_B + loss_fake_B).mean()

            # Vzvratno razširjanje in posodobitev
            D_loss_B.backward()
            D_B_optim.step()

            # Dodajanje na seznam
            D_losses_A.append(D_loss_A.item())
            D_losses_B.append(D_loss_B.item())
            G_losses.append(G_loss.item())

            # Izpis statistike
            if (i+1) % 200 == 0:
                print("Epoha [{}/{}] | Iteracija [{}/{}] | D_A Izguba {:.4f} | D_B Izguba {:.4f} | G Izguba {:.4f}"
                      .format(epoch + 1, num_epoch, i + 1, total_batch, np.average(D_losses_A), np.average(D_losses_B), np.average(G_losses)))

                # Shranjevanje primerov slik
                sample_images(test_satellite_loader, test_maps_loader, G_A2B, G_B2A, epoch, './results/samples/')

        # Prilagajanje stopnje učenja
        D_A_optim_scheduler.step()
        D_B_optim_scheduler.step()
        G_optim_scheduler.step()

        # Shranjevanje uteži modela
        if (epoch+1) % 10 == 0:
            torch.save(G_A2B.state_dict(), os.path.join('./results/weights/', 'CycleGAN_Generator_A2B_Epoch_{}.pkl'.format(epoch+1)))
            torch.save(G_B2A.state_dict(), os.path.join('./results/weights/', 'CycleGAN_Generator_B2A_Epoch_{}.pkl'.format(epoch+1)))

    print("Učenje končano.")


def inference():

    # Poti za sklepanje - generiranje
    paths = ['./results/inference/SatelliteToMap/', './results/inference/MapToSatellite/']
    paths = [make_dirs(path) for path in paths]

    # Priprava Dataloaderja
    test_satellite_loader, test_maps_loader = get_maps_loader('test', 1)

    # Priprava generatorja
    G_A2B = Generator().to(device)
    G_B2A = Generator().to(device)

    G_A2B.load_state_dict(torch.load(os.path.join('./results/weights/', 'CycleGAN_Generator_A2B_Epoch_{}.pkl'.format(num_epoch))))
    G_B2A.load_state_dict(torch.load(os.path.join('./results/weights/', 'CycleGAN_Generator_B2A_Epoch_{}.pkl'.format(num_epoch))))

    # Test
    print("Ustvarjanje slik...")
    for i, (satellite, maps) in enumerate(zip(test_satellite_loader, test_maps_loader)):

        # Priprava podatkov
        real_A = satellite.to(device)
        real_B = maps.to(device)

        # Generiranje lažnih slik
        fake_B = G_A2B(real_A)
        fake_A = G_B2A(real_B)

        # Generiranje rekonstruiranih slik
        fake_ABA = G_B2A(fake_B)
        fake_BAB = G_A2B(fake_A)

        # Shranjevanje slik (Satellite -> Map) #
        result = torch.cat((real_A, fake_B, fake_ABA), dim=0)
        save_image(denorm(result.data),
                   os.path.join('./results/inference/SatelliteToMap/', 'CycleGAN_SatelliteToMap_%03d.png' % (i+1)),
                   nrow=3,
                   normalize=True)

        # Shranjevanje slik (Map -> Satellite) #
        result = torch.cat((real_B, fake_A, fake_BAB), dim=0)
        save_image(denorm(result.data),
                   os.path.join('./results/inference/MapToSatellite/', 'CycleGAN_MapToSatellite_%03d.png' % (i+1)),
                   nrow=3,
                   normalize=True)


train()
inference()