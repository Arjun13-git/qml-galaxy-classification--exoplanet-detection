import os
import pandas as pd
import cv2
import numpy as np
import random
from tqdm import tqdm

# --- Configuration ---
RAW_IMG_DIR = '../data/raw_images' 
CSV_PATH = '../data/training_solutions_rev1.csv'
PROCESSED_DIR = '../data/processed_images' 

CLASSES = ['Elliptical', 'Spiral', 'Irregular']
for c in CLASSES:
    os.makedirs(os.path.join(PROCESSED_DIR, c), exist_ok=True)

# --- 1. Label Decoding & Filtering ---
print("Loading CSV and filtering for confident labels...")
df = pd.read_csv(CSV_PATH)

THRESHOLD = 0.80
IRREGULAR_THRESHOLD = 0.60 

ellipticals = df[df['Class1.1'] >= THRESHOLD]['GalaxyID'].astype(str).tolist()
spirals = df[(df['Class1.2'] >= THRESHOLD) & (df['Class4.1'] >= THRESHOLD)]['GalaxyID'].astype(str).tolist()
irregulars = df[(df['Class6.1'] >= IRREGULAR_THRESHOLD) & (df['Class8.4'] >= IRREGULAR_THRESHOLD)]['GalaxyID'].astype(str).tolist()

TARGET_COUNT = 2000

print(f"Found {len(ellipticals)} Ellipticals, {len(spirals)} Spirals, and {len(irregulars)} Irregulars.")
print(f"Targeting {TARGET_COUNT} images per class. Augmenting minority classes if necessary...\n")

# --- 2. Image Processing & Augmentation Engine ---
def augment_image(img):
    """Applies a random valid astronomical transformation"""
    transform = random.choice(['orig', 'flip_h', 'flip_v', 'rot_90', 'rot_180', 'rot_270'])
    if transform == 'flip_h': return cv2.flip(img, 1)
    elif transform == 'flip_v': return cv2.flip(img, 0)
    elif transform == 'rot_90': return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif transform == 'rot_180': return cv2.rotate(img, cv2.ROTATE_180)
    elif transform == 'rot_270': return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def process_and_save(galaxy_ids, class_name, target_count):
    saved_count = 0
    pbar = tqdm(total=target_count, desc=f"Processing {class_name}")
    
    while saved_count < target_count:
        for gal_id in galaxy_ids:
            if saved_count >= target_count:
                break
                
            img_path = os.path.join(RAW_IMG_DIR, f"{gal_id}.jpg")
            if not os.path.exists(img_path):
                continue
                
            img = cv2.imread(img_path)
            
            # 1. Wider Center Crop (300x300) to capture spiral arms
            center_x, center_y = 212, 212
            crop_size = 150 
            cropped_img = img[center_y-crop_size : center_y+crop_size, center_x-crop_size : center_x+crop_size]
            
            # 2. Anti-aliasing blur to prevent harsh pixelation
            blurred_img = cv2.GaussianBlur(cropped_img, (3, 3), 0)
            
            # 3. Resize to 64x64
            resized_img = cv2.resize(blurred_img, (64, 64), interpolation=cv2.INTER_AREA)
            
            if saved_count >= len(galaxy_ids):
                resized_img = augment_image(resized_img)
            
            # 4. Save as lossless PNG instead of JPG
            save_path = os.path.join(PROCESSED_DIR, class_name, f"{gal_id}_aug_{saved_count}.png")
            cv2.imwrite(save_path, resized_img)
            
            saved_count += 1
            pbar.update(1)
            
    pbar.close()

# --- 3. Execution ---
process_and_save(ellipticals, 'Elliptical', TARGET_COUNT)
process_and_save(spirals, 'Spiral', TARGET_COUNT)
process_and_save(irregulars, 'Irregular', TARGET_COUNT)

print("\nPhase 1 Preprocessing Complete! We now have a balanced dataset of 6,000 images.")