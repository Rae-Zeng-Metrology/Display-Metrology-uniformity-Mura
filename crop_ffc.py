import numpy as np
import cv2 
import os 

# 1) Input file path 
# -----------------------
FLAT_PATH ='flat_reference_gray_50pct.png'
RAW_PHOTO_PATH = [
'full_screen_gray_0pct.png',
'full_screen_gray_10pct.png',
'full_screen_gray_50pct.png',
'full_screen_gray_100pct.png',
]

OUT_DIR = "output_crop_fcc"
os.makedirs(OUT_DIR,exist_ok = True)

# 2) ROI
# -----------------------
# ROI = (x1, y1, x2, y2)
ROI = (510, 510, 3300, 2250)

def load_gray(path:str) -> np.ndarray:
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError (f"Cannot find image in: {path}")
    return img
    
def crop(img:np.ndarray,roi):
    x1,y1,x2,y2 = roi
    return img[y1:y2,x1:x2]
    
def ffc(raw_crop:np.ndarray,flat_crop:np.ndarray) -> np.ndarray:
    """
    Flat-field correction:
      corrected = raw / flat
      then normalize to mean=1 for convenient "relative luminance" interpretation
    """
    raw_f = raw_crop.astype(np.float32)
    flat_f = flat_crop.astype(np.float32)
    
    """
    Clip pixels that non-physical near 0 intensity to the threshold value to avoid error when devide
    """
    eps = 1e-6
    flat_f = np.clip(flat_f,eps, None)
    
    corrected = raw_f/flat_f
    corrected = corrected/np.mean(corrected) # bring to mean = 1 scale
    
    return corrected
    
def save_debug_images(name:str, raw_crop:np.ndarray,flat_crop:np.ndarray,corrected:np.ndarray):
    # 1) Save cropped raw/flat（8-bit）
    cv2.imwrite(os.path.join(OUT_DIR,f"{name}_raw_crop.png"),raw_crop)
    cv2.imwrite(os.path.join(OUT_DIR,f"{name}_flat_crop.png"),flat_crop)
    
    # 2) Save corrected（float）as npy
    np.save(os.path.join(OUT_DIR,f"{name}_corrected.npy"),corrected)
    
    # 3) Generate heatmap for visulization （Project 0.8~1.2 as 0~255）
    vis = np.clip((corrected-0.8)/(1.2-0.8),0,1)
    vis_u8 = (vis*255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT_DIR,f"{name}_corrected_vis.png"),vis_u8)
     
def main():
    print("loading flat reference image:", FLAT_PATH)
    flat = load_gray(FLAT_PATH)
    
    x1,y1,x2,y2 = ROI
    H,W = flat.shape
    if not (0 <= x1<x2 <= W and 0 <= y1 < y2 <= H):
        raise ValueError(f"{ROI}is out of bound for flat field reference image size {W}*{H}")
        
    flat_crop = crop(flat,ROI)
    print("flat field reference image size:", flat.shape,"Flat_crop size:",flat_crop.shape)
    
    # Save flat_crop 
    cv2.imwrite(os.path.join(OUT_DIR,"flat_crop_only.png"),flat_crop)
    print("Saved:",os.path.join(OUT_DIR,"flat_crop_only.png"))
    
    if len(RAW_PHOTO_PATH) == 0:
        print("RAW_PHOTO_PATH is empty")
        return
        
    for raw_path in RAW_PHOTO_PATH:
        print("Processing:",raw_path)
        raw = load_gray(raw_path)
        
        if raw.shape != flat.shape:
            raise ValueError( f"Raw image size {raw.shape} != flat image size {flat.shape}.")
            
        raw_crop =  crop(raw,ROI)
        corrected = ffc(raw_crop,flat_crop)
        
        base = os.path.splitext(os.path.basename(raw_path))[0]
        save_debug_images(base, raw_crop, flat_crop, corrected)
        print("Saved outputs for:", base)

    print("\nDone. Outputs in:", OUT_DIR)
    
    
if __name__ == "__main__":
    main()
            
            
    

    
