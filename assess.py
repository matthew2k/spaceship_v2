import torch
import numpy as np

from helpers import make_data, score_iou
from train_2 import SpaceshipDetector  # or wherever your model class is defined


def evaluate_model(model_path="model.pt", num_samples=100):
    """
    Loads the saved model, generates a batch of data, runs inference,
    and computes the average IoU.
    """
    # 1. Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpaceshipDetector(image_size=200, base_filters=8)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    iou_scores = []
    
    for _ in range(num_samples):
        # 2. Generate a random sample
        #    The default make_data has has_spaceship=None => 80% likely ship
        img, label = make_data(has_spaceship=None)
        
        # Convert to tensor [1, 1, H, W]
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # 3. Model prediction
        with torch.no_grad():
            pred = model(img_tensor)  # shape: (1, 5)
        pred = pred.squeeze(0).cpu().numpy()  # shape: (5,)
        
        # 4. Compute IoU
        iou = score_iou(pred, label)
        
        # score_iou returns None for true negative (both no ship).
        # We'll treat None as a "perfect result" for no-ship images 
        # or skip them from the average. You can adapt as needed.
        if iou is not None:
            iou_scores.append(iou)

    # 5. Print results
    if len(iou_scores) > 0:
        mean_iou = np.mean(iou_scores)
        print(f"Average IoU over {len(iou_scores)} samples: {mean_iou:.4f}")
    else:
        print("No positive (spaceship-present) samples encountered, cannot compute IoU.")


if __name__ == "__main__":
    evaluate_model(model_path="model.pt", num_samples=1000)
