import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import wandb

wandb.init(project="simple-gan-project", name="latent-interpolation", job_type="inference")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 100

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
        z = z.view(z.size(0), latent_dim, 1, 1)
        img = self.model(z)
        return img

generator = Generator(latent_dim).to(device)
generator.load_state_dict(torch.load("generator_final.pth", map_location=device))
generator.eval()

def interpolate(z1, z2, steps):
    vectors = []
    for alpha in torch.linspace(0, 1, steps):
        vec = (1 - alpha) * z1 + alpha * z2
        vectors.append(vec.unsqueeze(0))
    return torch.cat(vectors, dim=0)

z1 = torch.randn(latent_dim, device=device)
z2 = torch.randn(latent_dim, device=device)

steps = 20
interpolated_noise = interpolate(z1, z2, steps)

with torch.no_grad():
    fake_images = generator(interpolated_noise).detach().cpu()

grid = vutils.make_grid(fake_images, nrow=steps, padding=2, normalize=True)

plt.figure(figsize=(20, 5))
plt.axis("off")
plt.title("Latent Space Interpolation")
plt.imshow(grid.permute(1, 2, 0))
plt.savefig("interpolation.png")
plt.close()  

wandb.log({"interpolation_grid": wandb.Image("interpolation.png")})
wandb.finish()
