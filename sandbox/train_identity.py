# sandbox/train_identity.py
import os, sys, time
import torch
import torch.nn as nn
import torch.optim as optim

# Add repo root so the import works no matter where you run from
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from UNet_Surrogate_Model.models.unet_pbc_prelu import UNetPBCPrelu as Model

# --- Tunables for speed/verbosity ---
STEPS        = 200     # was 800; quicker for a first pass
BATCH_SIZE   = 4       # was 8
H = W        = 128     # was 256
PRINT_EVERY  = 10

def make_batch(bs=BATCH_SIZE, H=H, W=W, device="cpu"):
    """Synthetic pairs where target == input, so the model learns identity."""
    # x shape: [bs, 2, H, W]
    x = torch.zeros(bs, 2, H, W, device=device)
    # channel 0: sparse binary "vessels" (~10% ones)
    x[:, 0] = (torch.rand(bs, H, W, device=device) > 0.9).float()
    # channel 1: smoother VEGF field (average a few uniforms)
    v = sum(torch.rand(bs, H, W, device=device) for _ in range(4)) / 4.0
    x[:, 1] = v
    return x  # both input and target are x

def main():
    # Use MPS on Apple Silicon if available; fallback to CPU otherwise
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    model = Model().to(device)
    # BCEWithLogitsLoss: compare raw logits to targets in [0,1] without sigmoid
    loss_fn = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=3e-4)

    model.train()
    t0 = time.time()
    for step in range(1, STEPS + 1):
        x = make_batch(device=device)               # [B, 2, H, W]
        logits = model(x)                           # [B, 2, H, W] (logits)
        loss = loss_fn(logits, x)                   # target == input (identity)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % PRINT_EVERY == 0 or step == 1:
            print(f"[{step:04d}/{STEPS}] loss={loss.item():.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/unet_identity.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"saved {ckpt_path} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    # Optional: allow MPS->CPU fallback if something isn’t implemented on your chip
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
