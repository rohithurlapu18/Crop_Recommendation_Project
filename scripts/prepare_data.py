# ===========================================
# Data Preparation Script for Crop Recommendation Project
# ===========================================

import pandas as pd
import os

# Define input and output paths
input_path = os.path.join('data', 'raw', 'Crop_recommendation.csv')
output_path = os.path.join('data', 'processed', 'crop_clean.csv')

# Load dataset
print("🔹 Loading dataset...")
df = pd.read_csv(input_path)
print("✅ Dataset loaded successfully!\n")

# Display basic info
print("🔸 First 5 rows of data:")
print(df.head(), "\n")

# Check shape and null values
print(f"Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")
print("\n🔸 Checking for missing values:")
print(df.isnull().sum(), "\n")

# Remove duplicates
before = df.shape[0]
df.drop_duplicates(inplace=True)
after = df.shape[0]
print(f"✅ Removed {before - after} duplicate rows.\n")

# Describe statistics
print("🔹 Dataset statistics:")
print(df.describe(), "\n")

# Save cleaned dataset
os.makedirs('data/processed', exist_ok=True)
df.to_csv(output_path, index=False)
print(f"✅ Cleaned dataset saved to: {output_path}")
