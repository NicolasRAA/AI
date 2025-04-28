import torch
import torchvision.utils as vutils
import gradio as gr
import wandb
from interpolate import Generator 

wandb.init(project="simple-gan-project", name="grid-generation", job_type="inference")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 100

generator = Generator(latent_dim).to(device)
generator.load_state_dict(torch.load("generator_final.pth", map_location=device))
generator.eval()

def generate_grid(seed):
    torch.manual_seed(seed)
    num_images = 25
    noise = torch.randn(num_images, latent_dim, device=device)
    
    with torch.no_grad():
        fake_images = generator(noise).cpu()

    fake_images = (fake_images + 1) / 2  # Desnormalizar
    grid = vutils.make_grid(fake_images, nrow=5, padding=2, normalize=False)

    wandb.log({f"Grid_Seed_{seed}": [wandb.Image(grid)]})

    return grid.permute(1, 2, 0).numpy()

demo = gr.Interface(
    fn=generate_grid,
    inputs=gr.Slider(0, 1000, value=0, step=1, label="Seed for Noise"),
    outputs="image",
    title="Anime Face Generator (5x5 Grid Mode)",
    description="Move the slider to generate different Anime Faces (5x5 grid) using the GAN model."
)

if __name__ == "__main__":
    demo.launch(share=True)
