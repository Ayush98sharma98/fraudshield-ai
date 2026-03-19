import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def train_and_save():
    if os.path.exists("models/fraud_model.pkl"):
        print("✅ Model already exists!")
        return

    print("🚀 Training model...")
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv("data/creditcard.csv")

    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    df['Time']   = scaler.fit_transform(df[['Time']])

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                    random_state=42, n_jobs=-1)
    model.fit(X_train_res, y_train_res)

    with open("models/fraud_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("✅ Model saved!")

if __name__ == "__main__":
    train_and_save()
# ```

# ---

# ### Step 5 — Deploy on Streamlit Cloud

# 1. Go to **[share.streamlit.io](https://share.streamlit.io)**
# 2. Sign in with **GitHub**
# 3. Click **"New app"**
# 4. Fill in:
# ```
# Repository  : YOURUSERNAME/fraudshield-ai
# Branch      : main
# Main file   : app.py