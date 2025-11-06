# sandbox/compare_input_output.py
import imageio.v2 as imageio
import numpy as np
from pathlib import Path

def mse(a,b): 
    a = a.astype(np.float32)/255.0
    b = b.astype(np.float32)/255.0
    return float(((a-b)**2).mean())

def mad(a,b): 
    a = a.astype(np.float32)/255.0
    b = b.astype(np.float32)/255.0
    return float(np.abs(a-b).mean())

p = Path("out")
im_in_m  = imageio.imread(p/"input_mask.png")
im_out_m = imageio.imread(p/"next_mask.png")
im_in_v  = imageio.imread(p/"input_vegf.png")
im_out_v = imageio.imread(p/"next_vegf.png")

print("MASK  MSE:", mse(im_in_m, im_out_m), "  MAD:", mad(im_in_m, im_out_m))
print("VEGF  MSE:", mse(im_in_v, im_out_v), "  MAD:", mad(im_in_v, im_out_v))
