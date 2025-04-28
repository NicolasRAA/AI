import os
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import wandb

if __name__ == "__main__":
    wandb.init(project="simple-gan-project", name="dcgan-trainingver-v3")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU detected:", torch.cuda.get_device_name(0))
    else:
        print("GPU no detectada, se usará CPU")

    # (3) Corrección de transforms
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = datasets.ImageFolder(root="./anime_faces", transform=transform)

    latent_dim = 100
    num_epochs = 75
    batch_size = 128
    lr_G = 0.0002
    lr_D = 0.0001

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    class Generator(nn.Module):
        def __init__(self, latent_dim):
            super(Generator, self).__init__()
            self.model = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        def forward(self, z):
            z = z.view(z.size(0), latent_dim, 1, 1)  # (3) reshape explícito
            img = self.model(z)
            return img

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, img):
            validity = self.model(img)
            return validity.view(-1, 1)  # (3) reshape explícito

    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_G, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999))

    adversarial_loss = nn.MSELoss()

    fixed_noise = torch.randn(64, latent_dim, device=device)

    def save_generated_images(epoch, fixed_noise, generator, device):
        with torch.no_grad():
            fake_images = generator(fixed_noise.to(device)).detach().cpu()
        grid = vutils.make_grid(fake_images, padding=2, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title(f"Generated Images at Epoch {epoch}")
        plt.imshow(grid.permute(1, 2, 0))
        plt.savefig(f"generated_epoch_{epoch}.png")  # (3) solo guardar, no mostrar
        plt.close()
        wandb.log({f"Generated Images at Epoch {epoch}": [wandb.Image(grid)]})

    for epoch in range(1, num_epochs + 1):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size_curr = real_images.size(0)
            real_images = real_images.to(device)

            optimizer_D.zero_grad()

            real_labels = torch.full((batch_size_curr, 1), 0.9, device=device)
            fake_labels = torch.full((batch_size_curr, 1), 0.1, device=device)

            noise = torch.randn(batch_size_curr, latent_dim, device=device)
            fake_images = generator(noise)

            outputs_real = discriminator(real_images)
            outputs_fake = discriminator(fake_images.detach())

            d_loss_real = adversarial_loss(outputs_real, real_labels)
            d_loss_fake = adversarial_loss(outputs_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # (3) Actualizar Generator 3 veces
            for _ in range(3):
                optimizer_G.zero_grad()
                noise = torch.randn(batch_size_curr, latent_dim, device=device)
                fake_images = generator(noise)
                outputs = discriminator(fake_images)
                g_loss = adversarial_loss(outputs, real_labels)
                g_loss.backward()
                optimizer_G.step()

            wandb.log({
                "Generator Loss": g_loss.item(),
                "Discriminator Loss": d_loss.item()
            })

        if epoch % 5 == 0 or epoch == 1:
            save_generated_images(epoch, fixed_noise, generator, device)

        print(f"Epoch [{epoch}/{num_epochs}] | Generator Loss: {g_loss.item():.4f} | Discriminator Loss: {d_loss.item():.4f}")

    # (3) Guardar modelo entrenado
    torch.save(generator.state_dict(), "generator_final.pth")
    torch.save(discriminator.state_dict(), "discriminator_final.pth")
    wandb.finish()
