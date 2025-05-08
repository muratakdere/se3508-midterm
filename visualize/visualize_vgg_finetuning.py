import sys
import os
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import models
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.vgg16(pretrained=True)
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(25088, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, 5)
)
model.load_state_dict(torch.load("saved_models/vgg16_finetuning.pth", map_location=device))
model.to(device)
model.eval()

# to store activations
activation_maps = {}

def get_activation(name):
    def hook(model, input, output):
        activation_maps[name] = output.detach()
    return hook

# register hooks
model.features[0].register_forward_hook(get_activation("conv1"))
model.features[5].register_forward_hook(get_activation("conv3"))
model.features[10].register_forward_hook(get_activation("conv5"))

# select a random image from data/train
def get_random_image_path(root_dir="data/train"):
    class_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    selected_class = random.choice(class_dirs)
    image_files = [f for f in os.listdir(selected_class) if f.endswith(('.jpg', '.png', '.jpeg'))]
    selected_image = random.choice(image_files)
    return os.path.join(selected_class, selected_image)


image_path = get_random_image_path()
image = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0).to(device)

# normalize tensor values to the [0, 1] range for visualization purposes
def normalize_tensor(tensor):
    tensor = tensor - tensor.min()
    tensor = tensor / (tensor.max() + 1e-5)
    return tensor

# show original image + feature maps side by side
def visualize_feature_map(act_map, layer_name, num_filters=6, save_dir="saved_images/vgg_finetuning"):
    os.makedirs(save_dir, exist_ok=True)

    # upsample feature maps to the original image size (224x224)
    upsampled = F.interpolate(act_map, size=(224, 224), mode='bilinear', align_corners=False)
    num_filters = min(num_filters, upsampled.shape[1])

    # convert the original image to numpy format
    orig_display = transforms.ToTensor()(image.resize((224, 224)))
    orig_display = orig_display.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, num_filters + 1, figsize=(4 * (num_filters + 1), 5))

    
    axes[0].imshow(orig_display)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # feature maps
    for i in range(num_filters):
        fmap = normalize_tensor(upsampled[0, i]).cpu().numpy()
        axes[i + 1].imshow(fmap, cmap="inferno")
        axes[i + 1].axis("off")
    plt.suptitle(f"Original + Feature Maps - {layer_name}")

    save_path = os.path.join(save_dir, f"{layer_name}.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.show()

# to perform forward pass
with torch.no_grad():
    _ = model(input_tensor)


for layer in ["conv1", "conv3", "conv5"]:
    visualize_feature_map(activation_maps[layer], layer)




