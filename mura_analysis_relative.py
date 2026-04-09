"""
mura_analysis_relative.py

Purpose
-------
Given corrected.npy (relative luminance map, mean~1):
1) Compute low-frequency component via Gaussian low-pass filter.
2) Compute mura map = low_freq - 1.0
3) Report RMS mura and Peak-to-Peak mura.
4) Save heatmaps with colorbars + save float arrays (.npy).

Inputs
------
- corrected.npy: 2D float array, relative luminance, mean~1
Outputs
-------
- out_mura/low_freq_rel_heatmap.png
- out_mura/mura_map_heatmap.png
- out_mura/low_freq_rel.npy, mura_map.npy
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ==============================
# SETTINGS
# ==============================
CORRECTED_NPY_PATH = r"output_crop_fcc\full_screen_gray_100pct_corrected.npy"

OUT_DIR = "out_mura"
os.makedirs(OUT_DIR,exist_ok = True)

# Low-pass sigma (pixels): larger => more "cloudy" low-frequency mura
GAUSS_SIGMA = 60  # typical 30~100

# Optional: exclude border pixels (avoid edge roll-off/crop boundary artifacts)
BORDER_EXCLUDE_PX = 50

# Visualization windows (relative)
LOWFREQ_VMIN, LOWFREQ_VMAX = 0.90, 1.10
MURA_VMIN, MURA_VMAX = -0.1, 0.1

LOWFREQ_CMAP = "viridis"
MURA_CMAP = "seismic"

# ==============================
# Helpers
# ==============================
def load_corrected_relative(path:str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError (f"Expected 2D array, got shape ={arr.shape}")
    arr = arr.astype(np.float32)
    return arr
    
def crop_border(arr:np.ndarray,b:int) -> np.ndarray:
    if b <=0:
        return arr
    if arr.shape[0] <= 2*b or arr.shape[1] <= 2*b:
        raise ValueError("BORDER_EXCLUDE_PX too large for this image.")
    return arr[b:-b,b:-b]
    
def save_heatmap_with_colorbar(
    data: np.ndarray,
    out_path:str,
    vmin:float,
    vmax:float,
    cmap: str,
    title:str,
    cbar_label:str,
):
    im = plt.imshow(data,vmin = vmin, vmax = vmax,cmap = cmap)
    plt.title(title)
    plt.axis("off")
    plt.colorbar()
    plt.savefig(out_path,dpi = 300, bbox_inches='tight')
    plt.close()
    
# ==============================
# Main
# ==============================
def main():
    lum_rel = load_corrected_relative(CORRECTED_NPY_PATH)
    lum_eval = crop_border(lum_rel,BORDER_EXCLUDE_PX)
    
    # Low-frequency luminance (Gaussian low-pass)
    low_freq = cv2.GaussianBlur(lum_eval,ksize = (0,0), sigmaX = GAUSS_SIGMA, sigmaY = GAUSS_SIGMA)
    
    # Mura map: low-frequency deviation from ideal (=1)
    mura_map = (low_freq -1).astype(np.float32)
    
    # Mura metrics 
    rms = float(np.std(mura_map))
    p2p = float(np.max(mura_map)-np.min(mura_map))
    
    print("=== Mura Metrics (Low-frequency) ===")
    print(f"GAUSS_SIGMA: {GAUSS_SIGMA} px")
    print(f"RMS mura: {rms:.6f} ({rms*100:.3f}%)")
    print(f"P2P mura: {p2p:.6f} ({p2p*100:.3f}%)")
    
    # Save arrays
    np.save(os.path.join(OUT_DIR,"luminance_rel_used.npy"),lum_eval)
    np.save(os.path.join(OUT_DIR,"low_freq_rel.npy"),low_freq)
    np.save(os.path.join(OUT_DIR,"mura_map.npy"),mura_map)
    
    # Save heatmaps with colorbars
    save_heatmap_with_colorbar(
        low_freq,
        out_path = os.path.join(OUT_DIR,"low_freq_rel_heatmap.png"),
        vmin = LOWFREQ_VMIN,
        vmax = LOWFREQ_VMAX,
        cmap = LOWFREQ_CMAP,
        title = "Low-frequency Relative Luminance",
        cbar_label = "Relative luminance (mean = 1)",
    )
    
    save_heatmap_with_colorbar(
        mura_map,
        out_path = os.path.join(OUT_DIR,"mura_map_heatmap.png"),
        vmin = MURA_VMIN,
        vmax = MURA_VMAX,
        cmap = MURA_CMAP,
        title = "Low-frequency Mura Map",
        cbar_label = "Relative Deviation",
    )
    
    print(f"All output have been saved to: {OUT_DIR}")
    
    
if __name__ == "__main__":
    main()
        