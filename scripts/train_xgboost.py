# ======================================================
# XGBoost Crop Recommendation Model (Advanced Version)
# ======================================================

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load Cleaned Dataset
data_path = os.path.join('data', 'processed', 'crop_clean.csv')
print("🔹 Loading processed dataset...")
df = pd.read_csv(data_path)
print("✅ Dataset loaded successfully!\n")

# Step 2: Split features & labels
X = df.drop('label', axis=1)
y = df['label']

# Encode label values (XGBoost needs numeric labels)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Step 4: Define the XGBoost model
print("🌱 Training XGBoost model...")
xgb_model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.3,
    reg_lambda=2,
    random_state=42,
    tree_method='hist'
)

# Step 5: Train the model
xgb_model.fit(X_train, y_train)
print("✅ XGBoost model training completed!\n")

# Step 6: Evaluate the model
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Model Accuracy: {accuracy*100:.2f}%\n")

print("🔍 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Step 7: Cross-validation
cv_scores = cross_val_score(xgb_model, X, y_encoded, cv=5)
print(f"🔁 Cross-validation Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)\n")

# Step 8: Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.colorbar()
plt.show()

# Step 9: Save the model and label encoder
os.makedirs('models', exist_ok=True)
joblib.dump(xgb_model, 'models/xgboost_crop_model.joblib')
joblib.dump(label_encoder, 'models/label_encoder.joblib')
print("💾 Model and label encoder saved successfully!\n")

# Step 10: Test with a sample
sample = X_test.iloc[5].values.reshape(1, -1)
predicted_class = xgb_model.predict(sample)[0]
predicted_label = label_encoder.inverse_transform([predicted_class])[0]
print(f"🌾 Sample Prediction → Recommended Crop: {predicted_label}")
