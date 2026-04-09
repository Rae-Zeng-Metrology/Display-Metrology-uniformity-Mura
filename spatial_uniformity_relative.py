"""
spatial_uniformity_relative.py

Purpose
-------
1) Load FFC corrected relative luminance map (corrected.npy), enforce mean=1.
2) Compute luminance spatial uniformity metrics.
3) Compute color uniformity map: delta u'v' from an RGB image cropped to the same ROI.
4) Save heatmaps with colorbars.

Inputs
------
- corrected.npy: 2D float array, relative luminance map, mean ~ 1
- RGB image: must be cropped to SAME ROI as corrected.npy
  NOTE: OpenCV reads BGR. We convert BGR->RGB inside this script.

Outputs
-------
- out_uniformity/luminance_rel_heatmap.png
- out_uniformity/luminance_rel.npy
- out_uniformity/delta_upvp_heatmap.png, delta_upvp.npy
"""

import numpy as np
import cv2 
import os
import matplotlib.pyplot as plt

# ==============================
# USER SETTINGS (EDIT HERE)
# ==============================
CORRECTED_NPY_PATH = r"output_crop_fcc\full_screen_gray_100pct_corrected.npy"
BGR_IMG_PATH = r"output_crop_fcc\full_screen_gray_100pct_BGR_crop.png"

OUT_DIR = "uniformity"
os.makedirs(OUT_DIR,exist_ok = True)

# Luminance visualization window (relative, mean=1)
LUM_VMIN, LUM_VMAX = 0.8,1.2
LUM_CMAP = "viridis"

# For delta u'v': ignore too-dark pixels (reduces noisy chroma)
MIN_REL_LUM_FOR_COLOR = 0.2
DUV_VMIN, DUV_VMAX = 0.0, 0.01
DUV_CMAP = "inferno"

# Define eps for safe of mathatical deviding 
eps = 1e-12

# ==============================
# Helpers
# ==============================
def load_corrected_relative(path: str) -> np.ndarray:
    """Load corrected relative luminance (2D float) and enforce mean=1."""
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D corrected npy (H,W).Got shape = {arr.shape}")
    arr = arr.astype(np.float32)
    return arr/(np.mean(arr)+ eps)
    
def compute_luminance_uniformity_metrics(lum_rel: np.ndarray) -> dict:
    """
    lum_rel: relative luminance map with mean ~ 1.
    """
    mean_v = float(np.mean(lum_rel))
    min_v = float(np.min(lum_rel))
    max_v = float(np.max(lum_rel))
    std_v = float(np.std(lum_rel))
    non_uni = (max_v - min_v)/(mean_v + eps)
    
    return {
        "mean": mean_v,
        "min": min_v,
        "max": max_v,
        "stddev": std_v,
        "non_uniformity_percent": non_uni* 100.0,
    }
    
def save_heatmap_with_colorbar(
    data:np.ndarray,
    out_path:str,
    vmin: float,
    vmax:float,
    cmap:str,
    title:str,
    cbar_label: str,
):
    """Save a heatmap PNG with a colorbar (visualization only)."""
    im = plt.imshow(data, vmin = vmin, vmax = vmax, cmap = cmap)
    plt.title(title)
    plt.axis("off")
    plt.colorbar()
    plt.savefig(out_path, dpi = 300, bbox_inches='tight')
    plt.close()
        
def load_rgb_u8_from_cv2(path: str) -> np.ndarray:
    """
    Load image with OpenCV (BGR), then convert to RGB uint8.
    Output: rgb_u8 shape (H,W,3), dtype uint8, channel order RGB.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np. uint8)

# ---- Color conversion: RGB -> XYZ -> u'v' (engineering approximation) ----    
def srgb_u8_to_linear(rgb_u8: np.ndarray) -> np.ndarray:
     """uint8 RGB (0..255) -> linear RGB float (0..1)."""
     x = rgb_u8.astype(np.float32)/255.0
     return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
     
def srgb_linear_to_xyz_srgb(rgb_lin: np.ndarray) ->np.ndarray:
    """linear RGB -> XYZ using sRGB D65 matrix."""
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    return rgb_lin @ M.T
    
def xyz_to_upvp(XYZ: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """XYZ -> u'v'."""
    X = XYZ[..., 0]
    Y = XYZ[..., 1]
    Z = XYZ[..., 2]
    den = np.clip(X + 15.0 * Y + 3.0 * Z, eps, None)
    u_p = 4.0 * X / den
    v_p = 9.0 * Y / den
    return u_p, v_p
    
def compute_delta_upvp(rgb_u8:np.ndarray,lum_rel:np.ndarray) -> tuple[np.ndarray,dict]:
    """
    delta u'v' map:
      duv = sqrt((u-u_mean)^2 + (v-v_mean)^2)
    """
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError (f"rgb_u8 must be (H,W,3). Got {rgb_u8.shape}")
        
    rgb_lin = srgb_u8_to_linear (rgb_u8)
    XYZ = srgb_linear_to_xyz_srgb(rgb_lin)
    u,v = xyz_to_upvp(XYZ)
    
    mask = lum_rel > MIN_REL_LUM_FOR_COLOR
    if np.count_nonzero(mask) < 100: 
        raise ValueError ("Too few valid pixels after masking for color stats.")
        
    u_mean = float(np.mean(u[mask]))
    v_mean = float(np.mean(v[mask]))
    
    delta_uv = np.sqrt((u-u_mean)**2+(v-v_mean)**2).astype(np.float32)
    
    stats = {
        "u_mean":u_mean,
        "v_mean":v_mean,
        "delta_uv_mean": float(np.mean(delta_uv[mask])),
        "delta_uv_max": float(np.max(delta_uv[mask])),
        "delta_uv_p95": float(np.percentile(delta_uv[mask], 95)),
    }
    return delta_uv,stats
    
    
    
# ==============================
# Main
# ==============================
def main():
    # 1) Load corrected relative luminance
    lum_rel = load_corrected_relative(CORRECTED_NPY_PATH)
    
    # 2) Luminance uniformity metrics
    metrics = compute_luminance_uniformity_metrics(lum_rel)
    print("\n=== Spatial Uniformity: Relative Luminance (mean=1) ===")
    for k,v in metrics.items(): 
        print(f"{k}:{v}")
        
    # 3) Save luminance outputs
    np.save(os.path.join(OUT_DIR,"luminance_rel.npy"),lum_rel)
    
    save_heatmap_with_colorbar(
        lum_rel,
        os.path.join(OUT_DIR,"luminance_rel_heatmap.png"),
        vmin = LUM_VMIN,
        vmax = LUM_VMAX,
        cmap = LUM_CMAP,
        title = "Relative Luminance Uniformity",
        cbar_label = "Relative luminance (mean = 1)",
    )
    
    # 4) delta u'v'
    rgb_u8 = load_rgb_u8_from_cv2 (BGR_IMG_PATH) 
    if rgb_u8.shape[:2] != lum_rel.shape:
        raise ValueError(
            f"RGB shape {rgb_u8.shape[:2]} != luminance shape {lum_rel.shape}. "
            "Make sure RGB image is cropped to the same ROI as corrected.npy."
        )
            
    delta_uv,stats = compute_delta_upvp (rgb_u8,lum_rel)
    print("\n=== Spatial Uniformity: Δu′v′ ===")
    for k,v in stats.items():
        print(f"{k}: {v}")
    
    save_heatmap_with_colorbar(
        delta_uv,
        out_path=os.path.join(OUT_DIR, "delta_upvp_heatmap.png"),
        vmin=DUV_VMIN,
        vmax=DUV_VMAX,
        cmap=DUV_CMAP,
        title="Color Uniformity (Δu′v′)",
        cbar_label="Δu′v′",
    )
    
    print(f"\nAll outputs saved in: {OUT_DIR}")
    
    
    
if __name__ == "__main__":
    main()
    