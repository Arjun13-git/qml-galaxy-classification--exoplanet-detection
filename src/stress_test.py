import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from build_saan import LearnableNoiseGate, SEBlock

print("🔍 Initializing Hardware for Stress Test...")
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# --- 1. Load the Champion Model ---
print("🧠 Awakening SAAN-DenseNet121...")
MODEL_PATH = '../models/saan_densenet_best.keras'
model = tf.keras.models.load_model(
    MODEL_PATH, 
    custom_objects={'LearnableNoiseGate': LearnableNoiseGate, 'SEBlock': SEBlock}
)

# --- 2. Load the Data (Strict Validation Set) ---
CSV_PATH = '../master_labels_60k.csv'
IMAGES_DIR = r"C:\galaxy_datasets\images_training_rev1"

df = pd.read_csv(CSV_PATH)
df['filepath'] = IMAGES_DIR + '/' + df['filename']
df = df[df['filepath'].apply(os.path.exists)]

encoder = LabelEncoder()
df['label_encoded'] = encoder.fit_transform(df['Label'])

# We only want the 20% validation split so we don't test on images it already memorized
_, val_df = train_test_split(df, test_size=0.2, stratify=df['label_encoded'], random_state=42)

# Sample 2000 images to make the test run quickly
test_df = val_df.sample(n=2000, random_state=42)
num_classes = len(encoder.classes_)

# --- 3. The Deep Space Noise Generator ---
def add_gaussian_noise(image, snr):
    """Injects mathematical Gaussian static based on Signal-to-Noise Ratio"""
    # Calculate the variance of the clean image (the signal)
    signal_power = tf.math.reduce_variance(image)
    
    # Calculate how strong the noise needs to be
    noise_power = signal_power / snr
    
    # Generate the static
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=tf.math.sqrt(noise_power), dtype=tf.float32)
    
    # Apply static and clip the values so pixels stay between 0 and 1
    noisy_image = tf.clip_by_value(image + noise, 0.0, 1.0)
    return noisy_image

# --- 4. Testing Pipeline ---
def evaluate_at_snr(dataframe, snr_level, snr_name):
    print(f"\n🌌 Commencing {snr_name} Sweep...")
    
    def process_and_corrupt(file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (64, 64))
        img = img / 255.0 # DenseNet needs this!
        
        # Inject the cosmic radiation if it's not a clean run
        if snr_level is not None:
            img = add_gaussian_noise(img, snr_level)
            
        label = tf.one_hot(label, depth=num_classes)
        return img, label

    test_dataset = tf.data.Dataset.from_tensor_slices((dataframe['filepath'].values, dataframe['label_encoded'].values))
    test_dataset = test_dataset.map(process_and_corrupt, num_parallel_calls=tf.data.AUTOTUNE).batch(64)
    
    # Run the evaluation!
    results = model.evaluate(test_dataset, verbose=0)
    accuracy = results[1] * 100
    print(f"   => Accuracy: {accuracy:.2f}%")
    return accuracy

# --- 5. The Grand Evaluation ---
print("\n" + "="*50)
print("🚀 LAUNCHING SNR STRESS TEST (2,000 Images)")
print("="*50)

acc_clean = evaluate_at_snr(test_df, snr_level=None, snr_name="CLEAN (No Noise)")
acc_snr10 = evaluate_at_snr(test_df, snr_level=10.0, snr_name="SNR 10 (Light Static)")
acc_snr1 = evaluate_at_snr(test_df, snr_level=1.0, snr_name="SNR 1 (Heavy Static)")
acc_snr01 = evaluate_at_snr(test_df, snr_level=0.1, snr_name="SNR 0.1 (Severe Cosmic Radiation)")

print("\n" + "="*50)
print("📉 STRESS TEST RESULTS (THE CLASSICAL COLLAPSE)")
print("="*50)
print(f"Clean Data:    {acc_clean:.2f}%")
print(f"SNR 10:        {acc_snr10:.2f}%")
print(f"SNR 1:         {acc_snr1:.2f}%")
print(f"SNR 0.1:       {acc_snr01:.2f}%")
print("="*50)