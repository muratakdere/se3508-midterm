import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.custom_cnn import CustomCNN

# normalize tensor values to the [0, 1] range for visualization purposes
def normalize_tensor(tensor):
    tensor = tensor - tensor.min()
    tensor /= (tensor.max() + 1e-5)
    return tensor


def plot_feature_maps(original_img_tensor, feature_maps, layer_name, save_dir="saved_images/custom_cnn"):
    os.makedirs(save_dir, exist_ok=True)

    # upsample feature maps to the original image size (224x224)
    upsampled = F.interpolate(feature_maps, size=(224, 224), mode='bilinear', align_corners=False)
    num_filters = min(6, upsampled.shape[1])

    # convert the original image to numpy format
    original_img_np = original_img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(1, num_filters + 1, figsize=(4 * (num_filters + 1), 5))

    
    axes[0].imshow(original_img_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # feature maps
    for i in range(num_filters):
        fmap = normalize_tensor(upsampled[0, i]).detach().cpu().numpy()
        axes[i + 1].imshow(fmap, cmap='inferno')
        axes[i + 1].axis("off")

    plt.suptitle(f"Original + Feature Maps - {layer_name}")
    save_path = os.path.join(save_dir, f"{layer_name}.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()

# to store activations
activation_maps = {}

def get_activation(name):
    def hook(model, input, output):
        activation_maps[name] = output.detach()
    return hook

# capture and visualize feature maps
def visualize_feature_maps(model, device):
    model.eval()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(root='data/train', transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # to get a sample input
    inputs, _ = next(iter(loader))
    inputs = inputs.to(device)

    # register hooks 
    model.conv1.register_forward_hook(get_activation("conv1"))
    model.conv3.register_forward_hook(get_activation("conv3"))
    model.conv5.register_forward_hook(get_activation("conv5"))

    # to perform forward pass
    with torch.no_grad():
        _ = model(inputs)

    
    for layer in ["conv1", "conv3", "conv5"]:
        plot_feature_maps(inputs, activation_maps[layer], layer)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomCNN()
    model.load_state_dict(torch.load('saved_models/custom_cnn_best.pth', map_location=device))
    model.to(device)

    visualize_feature_maps(model, device)

if __name__ == "__main__":
    main()


