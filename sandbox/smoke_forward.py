# sandbox/smoke_forward.py
import os, sys, importlib, inspect
import torch, imageio

# ensure the repo root is on the path
sys.path.append(os.getcwd())

MODULE = "UNet_Surrogate_Model.models.unet_pbc_prelu"
C_IN, H, W = 2, 256, 256   # dataset.py shows 2 input channels

CANDIDATE_CLASSES = ["UNetPBCPrelu", "ResUNetPBCPrelu", "UNet"]

def get_model_class():
    mod = importlib.import_module(MODULE)
    for name in CANDIDATE_CLASSES:
        if hasattr(mod, name):
            print(f"Using {MODULE}.{name}")
            return getattr(mod, name)
    raise RuntimeError(f"No expected UNet class found in {MODULE}. "
                       f"Tried: {CANDIDATE_CLASSES}")

def dummy_state():
    x = torch.zeros(1, C_IN, H, W)
    x[:,0] = (torch.rand(1,1,H,W) > 0.85).float()  # fake vessel mask
    x[:,1] = torch.rand(1,1,H,W)                   # fake chemo field
    return x

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ModelClass = get_model_class()
    print("forward signature:", inspect.signature(ModelClass.forward))

    model = ModelClass().to(device).eval()
    x = dummy_state().to(device)
    with torch.no_grad():
        y = model(x)   # expect [1, 2, H, W]
    print("input  :", tuple(x.shape))
    print("output :", tuple(y.shape))

    os.makedirs("out", exist_ok=True)
    y0 = torch.sigmoid(y[0,0]).clamp(0,1).cpu().numpy()
    imageio.imwrite("out/unet_dummy.png", (y0*255).astype("uint8"))
    print("wrote out/unet_dummy.png")

if __name__ == "__main__":
    main()
