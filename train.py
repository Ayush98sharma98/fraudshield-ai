import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 50)
print("   CREDIT CARD FRAUD DETECTION - TRAINING")
print("=" * 50)

# ── Load Data ──────────────────────────────────────
df = pd.read_csv("data/creditcard.csv")
print(f"\n✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Fraud cases    : {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
print(f"   Normal cases   : {(df['Class'] == 0).sum()}")

# ── EDA Plots ──────────────────────────────────────
os.makedirs("plots", exist_ok=True)

# Class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df, palette=['#2ecc71', '#e74c3c'])
plt.title("Class Distribution (0=Normal, 1=Fraud)")
plt.xticks([0, 1], ['Normal', 'Fraud'])
plt.savefig("plots/class_distribution.png", dpi=150, bbox_inches='tight')
plt.close()

# Amount distribution
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
df[df['Class'] == 0]['Amount'].hist(bins=50, color='#2ecc71', alpha=0.7)
plt.title("Normal Transaction Amount")
plt.xlabel("Amount")
plt.subplot(1, 2, 2)
df[df['Class'] == 1]['Amount'].hist(bins=50, color='#e74c3c', alpha=0.7)
plt.title("Fraud Transaction Amount")
plt.xlabel("Amount")
plt.tight_layout()
plt.savefig("plots/amount_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ EDA plots saved in /plots")

# ── Preprocessing ──────────────────────────────────
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time']   = scaler.fit_transform(df[['Time']])

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── SMOTE ──────────────────────────────────────────
print("\n⚙️  Applying SMOTE to handle class imbalance...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"   Before SMOTE: {dict(y_train.value_counts())}")
print(f"   After  SMOTE: {dict(pd.Series(y_train_res).value_counts())}")

# ── Train Model ────────────────────────────────────
print("\n🚀 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_res, y_train_res)
print("✅ Model trained!")

# ── Evaluate ───────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
print(f"   ROC AUC Score : {roc_auc_score(y_test, y_prob):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig("plots/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Confusion matrix saved")

# Feature Importance
feat_imp = pd.Series(model.feature_importances_, index=X.columns)
top_features = feat_imp.nlargest(10)
plt.figure(figsize=(8, 5))
top_features.sort_values().plot(kind='barh', color='#3498db')
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Feature importance plot saved")

# ── Save Model & Scaler ────────────────────────────
os.makedirs("models", exist_ok=True)
with open("models/fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Model saved  → models/fraud_model.pkl")
print("✅ Scaler saved → models/scaler.pkl")
print("\n🎉 Training Complete! Now run: streamlit run app.py")