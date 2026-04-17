import os
import pandas as pd
import numpy as np
import tensorflow as tf
from build_saan import LearnableNoiseGate, SEBlock

print("🔍 Checking Hardware...")
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) == 0:
    raise RuntimeError("❌ NO GPU DETECTED! Aborting.")
else:
    print(f"✅ GPU Detected: {physical_devices[0]}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("🚀 Initializing Mixed Precision...")
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# --- 1. Load the Backlog ---
CSV_PATH = '../unclassified_backlog.csv'
IMAGES_DIR = r"C:\galaxy_datasets\images_training_rev1"

print("📂 Loading unclassified backlog...")
df = pd.read_csv(CSV_PATH)
df['filepath'] = IMAGES_DIR + '/' + df['filename']

# Verify files exist
df = df[df['filepath'].apply(os.path.exists)]
print(f"Valid backlog images ready for inference: {len(df)}")

# --- 2. The Inference Stream (No Augmentation!) ---
IMG_SIZE = (64, 64)
BATCH_SIZE = 64

def process_image_for_inference(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img

print("🌊 Building Data Stream...")
inference_dataset = tf.data.Dataset.from_tensor_slices(df['filepath'].values)
inference_dataset = inference_dataset.map(process_image_for_inference, num_parallel_calls=tf.data.AUTOTUNE)
inference_dataset = inference_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- 3. Load the Trained SAAN Model ---
print("🧠 Awakening the SAAN Model...")
MODEL_PATH = '../models/saan_best_model_60k.keras'
# We must pass the custom blocks so Keras knows how to reconstruct the architecture
model = tf.keras.models.load_model(
    MODEL_PATH, 
    custom_objects={'LearnableNoiseGate': LearnableNoiseGate, 'SEBlock': SEBlock}
)

# --- 4. The Grand Classification ---
print("⚡ Unleashing GPU for massive parallel inference...")
predictions = model.predict(inference_dataset)

# Scikit-Learn's LabelEncoder alphabetizes classes: 0=Elliptical, 1=Irregular, 2=Spiral
CLASS_NAMES = ['Elliptical', 'Irregular', 'Spiral']

print("📊 Compiling results and confidence scores...")
# Get the highest probability score for each image
df['Confidence_Score'] = np.max(predictions, axis=1)
# Map the winning index back to the text label
df['Predicted_Label'] = [CLASS_NAMES[idx] for idx in np.argmax(predictions, axis=1)]

# --- 5. Save the New Custom Dataset ---
output_path = '../pseudo_labeled_backlog.csv'

# We only save the filename, the new label, and how confident the AI was
final_df = df[['filename', 'Predicted_Label', 'Confidence_Score']]
final_df.to_csv(output_path, index=False)

print("\n" + "="*50)
print(f"✅ INFERENCE COMPLETE!")
print(f"💾 Results saved to {output_path}")
print("="*50)

# Let's see how many of each it found!
print("\nNew Dataset Distribution:")
print(final_df['Predicted_Label'].value_counts())

print("\nAverage Confidence per Class:")
print(final_df.groupby('Predicted_Label')['Confidence_Score'].mean())