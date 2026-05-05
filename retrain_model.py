"""
Retrain the RandomForest crop recommendation model with the current sklearn version.
Uses the standard Crop Recommendation Dataset.
"""
import pickle
import urllib.request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATASET_URL = "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-processed/crop_recommendation.csv"
LOCAL_CSV = "Data/crop_recommendation.csv"

print("Downloading crop recommendation dataset...")
try:
    urllib.request.urlretrieve(DATASET_URL, LOCAL_CSV)
    print(f"Dataset saved to {LOCAL_CSV}")
except Exception as e:
    print(f"Download failed: {e}")
    print("Trying fallback URL...")
    FALLBACK_URL = "https://raw.githubusercontent.com/AtharvaD1/precision-agriculture-using-machine-learning/main/Data/crop_recommendation.csv"
    try:
        urllib.request.urlretrieve(FALLBACK_URL, LOCAL_CSV)
        print(f"Dataset saved from fallback to {LOCAL_CSV}")
    except Exception as e2:
        print(f"Fallback also failed: {e2}")
        raise SystemExit("Could not download dataset. Please manually place crop_recommendation.csv in the Data/ folder.")

df = pd.read_csv(LOCAL_CSV)
print(f"Dataset loaded: {df.shape[0]} rows, columns: {list(df.columns)}")

# Expect columns: N, P, K, temperature, humidity, ph, rainfall, label
feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
target_col = 'label'

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training RandomForest model...")
model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model accuracy on test set: {accuracy*100:.2f}%")

model_path = "models/RandomForest.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to {model_path}")
print("Done! You can now run app.py.")
