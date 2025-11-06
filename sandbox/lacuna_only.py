# sandbox/lacuna_only.py
import imageio.v2 as imageio
import numpy as np

def main():
    img = imageio.imread("out/next_mask.png").astype(np.float32) / 255.0   # [0,1]
    mask = (img >= 0.5).astype(np.uint8)                                   # binarize
    vessel_fraction = mask.mean()                                          # proportion vessel
    lacuna_fraction = 1.0 - float(vessel_fraction)                         # empty area
    print("vessel_fraction:", float(vessel_fraction))
    print("lacuna_fraction:", lacuna_fraction)

if __name__ == "__main__":
    main()
