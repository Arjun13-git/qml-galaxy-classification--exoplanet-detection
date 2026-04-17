import pandas as pd
import numpy as np

print("🚀 Loading Labels CSV...")
# Hardcoded to the exact path you provided
labels_df = pd.read_csv(r'C:\galaxy_datasets\training_solutions_rev1.csv') 

def categorize_galaxy(row):
    """
    Strict thresholds for pristine data. Anything else gets flagged for the backlog.
    """
    if row['Class1.1'] >= 0.60:
        return 'Elliptical'
    elif row['Class1.2'] >= 0.60 and row['Class4.1'] >= 0.50:
        return 'Spiral'
    elif row['Class6.1'] >= 0.50 or row['Class8.4'] >= 0.50:
        return 'Irregular'
    else:
        return 'Unclassified'

print("🧠 Applying classification logic to survey data...")
labels_df['Label'] = labels_df.apply(categorize_galaxy, axis=1)

# Magic trick: Build the filename directly from the 6-digit GalaxyID
labels_df['filename'] = labels_df['GalaxyID'].astype(str) + '.jpg'

# --- The Split ---
# 1. The Pristine 60k -> Saving directly to your project root
clean_df = labels_df[labels_df['Label'] != 'Unclassified']
clean_csv_path = r'C:\Projects\QML_Galaxy_Project\master_labels_60k.csv'
clean_df[['filename', 'Label']].to_csv(clean_csv_path, index=False)

# 2. The Unclassified Backlog -> Saving directly to your project root
unclassified_df = labels_df[labels_df['Label'] == 'Unclassified']
backlog_csv_path = r'C:\Projects\QML_Galaxy_Project\unclassified_backlog.csv'
unclassified_df[['filename']].to_csv(backlog_csv_path, index=False)

print("\n" + "="*40)
print("✅ DATASET GENERATION COMPLETE")
print("="*40)
print(f"Total Original Images: {len(labels_df)}")
print(f"High-Confidence Images Kept: {len(clean_df)}")
print("\nClass Distribution for Training:")
print(clean_df['Label'].value_counts())
print("\n🚀 Next Step: Update train_saan.py with your image path!")