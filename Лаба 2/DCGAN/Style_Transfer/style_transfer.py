import torch
import torch.nn as nn
import torchvision.transforms as transforms
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import wandb
from interpolate import Generator  

wandb.init(project="simple-gan-project", name="style-transfer-v2", job_type="inference")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100

generator = Generator(latent_dim).to(device)
generator.load_state_dict(torch.load("generator_final.pth", map_location=device))
generator.eval()

transform_input = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def denormalize(tensor):
    tensor = (tensor + 1) / 2
    return tensor.clamp(0, 1)

# Función mejorada de transferencia de estilo
def style_transfer(img, mode, seed=None):
    img_tensor = transform_input(img).unsqueeze(0).to(device)

    if seed is not None:
        torch.manual_seed(seed)

    mse_loss = nn.MSELoss()

    if mode == "Fast transformation":
        noise = torch.randn(1, latent_dim, device=device)
        with torch.no_grad():
            output = generator(noise)
        output = denormalize(output).squeeze(0).permute(1, 2, 0).cpu().numpy()
        wandb.log({"Fast Transformation Result": [wandb.Image(output)]})
        return output

    elif mode == "Detailed transformation":
        best_img = None
        best_loss = float('inf')

        for _ in range(50):  
            noise = torch.randn(1, latent_dim, device=device)
            with torch.no_grad():
                candidate = generator(noise)

            candidate_denorm = denormalize(candidate)
            candidate_img = candidate_denorm.squeeze(0).unsqueeze(0).to(device)

            # MSE loss entre input y generado
            loss = mse_loss(candidate_img, img_tensor)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_img = candidate_denorm

        best_img = best_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        wandb.log({"Detailed Transformation Result": [wandb.Image(best_img)]})
        return best_img

# Gradio Interface
interface = gr.Interface(
    fn=style_transfer,
    inputs=[
        gr.Image(type="pil", label="Upload your image"),
        gr.Radio(choices=["Fast transformation", "Detailed transformation"], value="Fast transformation", label="Mode"),
        gr.Number(label="Optional Seed (leave empty for random)", value=None)
    ],
    outputs="image",
    title="Anime Style Transfer - Improved Version",
    description="Choose fast or detailed generation to stylize your image! Detailed tries much harder to match the input."
)

if __name__ == "__main__":
    interface.launch()
    wandb.finish()
