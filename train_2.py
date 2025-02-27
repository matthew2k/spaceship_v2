import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from helpers import make_data

# ---------------------------------------------------
# Simple PyTorch model replicating the structure
# from the given Keras code (train.py) but in PyTorch
# ---------------------------------------------------

class SpaceshipDetector(nn.Module):
    def __init__(self, image_size=200, base_filters=8):
        super().__init__()
        # We match the progression: [1, 2, 4, 8, 16, 32, 64],
        # multiplied by base_filters (8).
        filters_list = [base_filters * i for i in [1, 2, 4, 8, 16, 32, 64]]

        layers = []
        in_channels = 1  # single-channel input

        for out_channels in filters_list:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                    padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2))
            in_channels = out_channels

        self.conv_stack = nn.Sequential(*layers)
        
        # After 7 pooling operations on a 200x200 image, the spatial size
        # is approximately 200 / (2^7) = 1.5625, which floors to 1 in 
        # typical PyTorch MaxPool settings. So final feature map is [512, 1, 1].
        #  => 512 = base_filters*(64).
        
        # The final number of channels is filters_list[-1].
        # Flatten and predict 5 params (x, y, yaw, width, height).
        self.fc = nn.Linear(filters_list[-1], 5)

    def forward(self, x):
        # x shape: (batch, 1, 200, 200)
        feats = self.conv_stack(x)
        # feats shape likely: (batch, 512, 1, 1)
        feats = feats.view(feats.size(0), -1)  # flatten
        out = self.fc(feats)  # (batch, 5)
        return out


def make_batch(batch_size):
    """
    Generate a batch of training data where we always ensure a spaceship
    is present (like the provided train.py). If you wish to also train on
    the 'no spaceship' scenario, you'll need to adjust the data generation
    and possibly the model output.
    """
    imgs, labels = zip(*[make_data(has_spaceship=True) for _ in range(batch_size)])
    imgs = np.stack(imgs)     # shape: (batch, 200, 200)
    labels = np.stack(labels) # shape: (batch, 5)
    
    # Convert to torch tensors
    # We add a channel dimension (1) because our model expects (N,1,H,W).
    imgs = torch.from_numpy(imgs).float().unsqueeze(1)
    labels = torch.from_numpy(labels).float()
    return imgs, labels


def main():
    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model, optimizer, loss
    model = SpaceshipDetector(image_size=200, base_filters=8)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Optional: print a summary-like view of the model
    # A simple approach is to print the layers:
    print(model)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {param_count}")

    # Simple training loop generating data on-the-fly
    model.train()
    for epoch in range(EPOCHS):
        # In each epoch, we simulate steps_per_epoch (like in Keras example = 500).
        steps_per_epoch = 500
        epoch_loss = 0.0
        
        for step in range(steps_per_epoch):
            imgs, labels = make_batch(BATCH_SIZE)
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/steps_per_epoch:.4f}")

    # Save the final model weights
    torch.save(model.state_dict(), "model.pt")
    print("Training complete. Model saved to model.pt")


if __name__ == "__main__":
    main()
