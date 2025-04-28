import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os

import wandb

if __name__ == "__main__": 
    wandb.init(project="simple-gan-project", name="gan-training-after-changes-2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.ImageFolder(root="./anime_faces", transform=transform)

    latent_dim = 100
    # lr = 0.0001         #(1) 0.002 -> 0.0001       (2) Lo cambio dierectamente en cada optimizer para que el Discriminador entrene mas lento
    num_epochs = 75       #(2) 50 -> 75
    batch_size = 128

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


    class Generator(nn.Module):
        def __init__(self, latent_dim):
            super(Generator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(128, 256),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(256, 512),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(512, 1024),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(1024, 64*64*3),
                nn.Tanh()
            )

        def forward(self, z):
            img = self.model(z)
            img = img.view(z.size(0), 3, 64, 64)
            return img

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(3 * 64 * 64, 512),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(256, 1),
                nn.Sigmoid()
            )

        def forward(self, img):
            img_flat = img.view(img.size(0), -1)
            validity = self.model(img_flat)
            return validity

    fixed_noise = torch.randn(64, latent_dim, device=device)

    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))     #(1) (0.5, 0.999) -> (0.0, 0.9)     (2) dando lr mas rapido y devolviendo betas al valor anterior (0.0, 0.9) -> (0.5, 0.999)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999)) #(1) (0.5, 0.999) -> (0.0, 0.9)     (2) dando lr mas lento y devolviendo betas al valor anterior (0.0, 0.9) -> (0.5, 0.999)

    adversarial_loss = nn.BCELoss()

    def save_generated_images(epoch, fixed_noise):
        with torch.no_grad():
            fake_images = generator(fixed_noise).detach().cpu()
        grid = vutils.make_grid(fake_images, padding=2, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title(f"Generated Images at Epoch {epoch}")
        plt.imshow(grid.permute(1, 2, 0))
        plt.show()

        wandb.log({f"Generated Images at Epoch {epoch}": [wandb.Image(grid)]})

    for epoch in range(1, num_epochs + 1):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size_curr = real_images.size(0)
            real_images = real_images.to(device)

            # Discriminator
            optimizer_D.zero_grad()

            # Usare Label Smoothing (pag. 77)
            real_labels = torch.full((batch_size_curr, 1), 0.9, device=device, dtype=torch.float)     #(1) 1 -> 0.9
            fake_labels = torch.full((batch_size_curr, 1), 0.1, device=device, dtype=torch.float)     #(1) 0 -> 0.1

            outputs_real = discriminator(real_images)
            d_loss_real = adversarial_loss(outputs_real, real_labels)

            noise = torch.randn(batch_size_curr, latent_dim, device=device)
            fake_images = generator(noise)
            outputs_fake = discriminator(fake_images.detach())
            d_loss_fake = adversarial_loss(outputs_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # Generator
            optimizer_G.zero_grad()

            outputs = discriminator(fake_images)
            g_loss = adversarial_loss(outputs, real_labels)
            g_loss.backward()
            optimizer_G.step()

            wandb.log({
                "Generator Loss": g_loss.item(),
                "Discriminator Loss": d_loss.item()
            })

        if epoch % 5 == 0 or epoch == 1:
            save_generated_images(epoch, fixed_noise)

        print(f"Epoch [{epoch}/{num_epochs}] | Generator Loss: {g_loss.item():.4f} | Discriminator Loss: {d_loss.item():.4f}")
