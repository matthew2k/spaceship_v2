import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from helpers import make_data
from train_yaw import SpaceshipDetector6
from torchvision.utils import make_grid
import seaborn as sns
import sys

class LayerActivations:
    def __init__(self, model, layer_names):
        self.model = model
        self.activations = {}
        self.hooks = []
        
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Register hooks for each conv layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if name in layer_names:
                    self.hooks.append(
                        module.register_forward_hook(get_activation(name))
                    )

def visualize_feature_maps(model, img_tensor, layer_name):
    """Visualize feature maps for a specific layer"""
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # Get activations
    layer_viz = LayerActivations(model, [layer_name])
    model.eval()
    with torch.no_grad():
        _ = model(img_tensor)
    
    # Get the feature maps
    feature_maps = layer_viz.activations[layer_name]
    
    # Create grid of feature maps
    grid = make_grid(feature_maps[0], nrow=8, padding=2, normalize=True)
    plt.figure(figsize=(15, 15))
    plt.imshow(grid.cpu().numpy().transpose(1, 2, 0))
    plt.title(f'Feature Maps - {layer_name}')
    plt.axis('off')
    plt.show()

def analyze_activations(model, img_tensor):
    """Analyze activation statistics for all conv layers"""
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    conv_layers = [name for name, module in model.named_modules() 
                  if isinstance(module, nn.Conv2d)]
    
    layer_viz = LayerActivations(model, conv_layers)
    model.eval()
    with torch.no_grad():
        _ = model(img_tensor)
    
    stats = {}
    for name, activation in layer_viz.activations.items():
        act_data = activation.cpu().numpy()
        stats[name] = {
            'mean': float(np.mean(act_data)),
            'std': float(np.std(act_data)),
            'dead_filters': float((act_data == 0).mean()),
            'max': float(np.max(act_data))
        }
        
        # Create activation distribution plot
        plt.figure(figsize=(10, 4))
        sns.histplot(act_data.flatten(), bins=50)
        plt.title(f'Activation Distribution - {name}')
        plt.xlabel('Activation Value')
        plt.ylabel('Count')
        plt.show()
    
    return stats

def main():
    # Load the local model
    model = SpaceshipDetector6(image_size=200, base_filters=16)
    model.load_state_dict(torch.load("C:/Projects/spaceship_v2/model_yaw.pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Generate a few test images
    for i in range(5):
        img, label = make_data(has_spaceship=True, noise_level=0.3)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        
        # Show original image
        plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap='gray')
        plt.title(f'Original Image {i+1}')
        plt.axis('off')
        plt.show()
        
        # Visualize feature maps for each conv layer
        conv_layers = [name for name, module in model.named_modules() 
                      if isinstance(module, nn.Conv2d)]
        
        for layer_name in conv_layers:
            visualize_feature_maps(model, img_tensor, layer_name)
        
        # Analyze activations
        stats = analyze_activations(model, img_tensor)
        
        # Print summary statistics
        print(f"\nImage {i + 1} Analysis:")
        for layer, layer_stats in stats.items():
            print(f"\n{layer}:")
            print(f"  Mean activation: {layer_stats['mean']:.4f}")
            print(f"  Dead filters: {layer_stats['dead_filters']*100:.1f}%")
            print(f"  Max activation: {layer_stats['max']:.4f}")

if __name__ == "__main__":
    main()
