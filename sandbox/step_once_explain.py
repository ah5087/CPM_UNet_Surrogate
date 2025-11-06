# sandbox/step_once_explain.py
import os
import sys
import torch
import imageio.v2 as imageio
import numpy as np

"""
Runs a single forward pass of the UNet surrogate:
- Builds a simple 2-channel input: [mask, VEGF]
- Loads UNetPBCPrelu (and an identity checkpoint if present)
- Saves input and output images to ./out for quick inspection
"""

def to_png(arr: np.ndarray, path: str) -> None:
    """Normalize a 2D array to [0,1] and save as 8-bit PNG (helps visualize logits)."""
    arr = arr.astype(np.float32)
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    imageio.imwrite(path, (arr * 255).astype(np.uint8))

def main() -> None:
    # Ensure repo root (parent of sandbox/) is on sys.path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.abspath(os.path.join(script_dir, os.pardir))
    if repo_root not in sys.path:
        sys.path.append(repo_root)

    # Import model from the repo
    from UNet_Surrogate_Model.models.unet_pbc_prelu import UNetPBCPrelu as Model

    # Device: Apple Silicon MPS if available, else CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Build a toy current state: channel 0 = sparse mask, channel 1 = VEGF
    H = W = 256
    state = torch.zeros(1, 2, H, W)
    state[:, 0] = (torch.rand(1, H, W) > 0.9).float()  # ~10% vessel pixels
    state[:, 1] = torch.rand(1, H, W)                  # random VEGF in [0,1]
    state = state.to(device)

    # Construct model
    model = Model().to(device)

    # If you trained the quick identity model, load it; otherwise use random weights
    ckpt_path = os.path.join(repo_root, "checkpoints", "unet_identity.pt")
    if os.path.isfile(ckpt_path):
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"Loaded checkpoint from {ckpt_path}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint {ckpt_path}: {e}")
    else:
        print("No checkpoint found; using random weights.")

    model.eval()

    # Single forward pass; logits -> sigmoid for probabilities
    with torch.no_grad():
        logits = model(state)      # [1, 2, H, W]
        probs  = torch.sigmoid(logits)

    # Save inputs and outputs to ./out
    os.makedirs("out", exist_ok=True)
    to_png(state[0, 0].cpu().numpy(), "out/input_mask.png")
    to_png(state[0, 1].cpu().numpy(), "out/input_vegf.png")
    imageio.imwrite("out/next_mask.png", (probs[0, 0].cpu().numpy() * 255).astype(np.uint8))
    imageio.imwrite("out/next_vegf.png", (probs[0, 1].cpu().numpy() * 255).astype(np.uint8))
    print("Saved out/input_mask.png, out/input_vegf.png, out/next_mask.png, out/next_vegf.png")

if __name__ == "__main__":
    main()
