import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

# --- Configuration ---
# Make sure these paths match where you extracted your Kaggle files!
RAW_IMG_DIR = '../data/raw_images' 
CSV_PATH = '../data/training_solutions_rev1.csv'
PROCESSED_DIR = '../data/processed_images' 

# We will start with just Elliptical and Spiral for the initial baseline
CLASSES = ['Elliptical', 'Spiral']
for c in CLASSES:
    os.makedirs(os.path.join(PROCESSED_DIR, c), exist_ok=True)

# --- 1. Label Decoding & Filtering ---
print("Loading CSV and filtering for confident labels...")
df = pd.read_csv(CSV_PATH)

# Galaxy Zoo 1 Decision Tree mapping:
# Class1.1 = Smooth (Elliptical)
# Class1.2 = Features/Disk (Spiral/Lenticular)
# Class4.1 = Has Spiral Arms (Spiral)
THRESHOLD = 0.80

# Find strong Ellipticals
ellipticals = df[df['Class1.1'] >= THRESHOLD]['GalaxyID'].astype(str).tolist()

# Find strong Spirals (Has disk AND has spiral arms)
spirals = df[(df['Class1.2'] >= THRESHOLD) & (df['Class4.1'] >= THRESHOLD)]['GalaxyID'].astype(str).tolist()

print(f"Found {len(ellipticals)} strong Ellipticals and {len(spirals)} strong Spirals.")

# --- 2. Image Processing Function ---
def process_and_save_image(galaxy_id, class_name):
    img_path = os.path.join(RAW_IMG_DIR, f"{galaxy_id}.jpg")
    
    if not os.path.exists(img_path):
        return # Skip if image is missing from the batch
        
    # Read image
    img = cv2.imread(img_path)
    
    # Center Crop (Raw is 424x424, we crop the central 200x200 to remove black space)
    center_x, center_y = 212, 212
    crop_size = 100 # 100 pixels in each direction from center
    cropped_img = img[center_y-crop_size : center_y+crop_size, center_x-crop_size : center_x+crop_size]
    
    # Resize to 64x64 for the CNN
    resized_img = cv2.resize(cropped_img, (64, 64), interpolation=cv2.INTER_AREA)
    
    # Save to the respective class folder
    save_path = os.path.join(PROCESSED_DIR, class_name, f"{galaxy_id}.jpg")
    cv2.imwrite(save_path, resized_img)

# --- 3. Execution Loop with Progress Bar ---
# We will process 2000 of each to create a balanced dataset for our initial baseline
print(f"\nProcessing Ellipticals...")
for gal_id in tqdm(ellipticals[:2000]): 
    process_and_save_image(gal_id, 'Elliptical')

print(f"\nProcessing Spirals...")
for gal_id in tqdm(spirals[:2000]):
    process_and_save_image(gal_id, 'Spiral')

print("\nPhase 1 Preprocessing Complete! Images are cropped, resized, and sorted.")