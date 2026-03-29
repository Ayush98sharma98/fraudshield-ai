# 🛡️ FraudShield AI — Credit Card Fraud Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4.2-orange?style=for-the-badge&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge)

**An end-to-end Machine Learning system that detects fraudulent credit card transactions in real-time**

[🚀 Live Demo](https://fraudshield-ai-severus.streamlit.app) • [📊 Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) • [📓 Notebook](notebooks/fraud_detection.ipynb)

</div>

---

## 📌 Overview

FraudShield AI is a production-ready fraud detection system built on **284,807 real credit card transactions**. The key challenge was extreme class imbalance — only **0.17%** of transactions were fraudulent. This was solved using **SMOTE (Synthetic Minority Oversampling Technique)**, achieving a **99.5% accuracy** and **0.98 ROC AUC** score.

---

## 🎯 Results

| Metric | Score |
|--------|-------|
| ✅ Accuracy | 99.5% |
| 🎯 Precision | 98.7% |
| 🔍 Recall | 97.2% |
| ⚖️ F1 Score | 97.9% |
| 📈 ROC AUC | 0.980 |

---

## ✨ Features

- 🔍 **Live Detector** — Analyze single transactions in real-time with fraud risk gauge
- 📁 **Batch Scanner** — Upload CSV to scan thousands of transactions at once
- 📊 **Analytics Dashboard** — ROC curve, confusion matrix, feature importance
- 🎨 **6 Themes** — Cyber Dark, Ocean, Neon Ember, Violet Storm, Matrix Green, Arctic White
- 🧪 **Preset Test Cases** — Test fraud vs normal with one click
- ⬇️ **Download Results** — Export batch scan results as CSV

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.11 |
| **ML Framework** | Scikit-learn |
| **Imbalance Handling** | Imbalanced-learn (SMOTE) |
| **Web App** | Streamlit |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Data Processing** | Pandas, NumPy |

---

## 🔄 ML Pipeline

```
Raw Data (284K transactions)
        ↓
   Data Cleaning & EDA
        ↓
   StandardScaler (Amount & Time)
        ↓
   Train-Test Split (80:20, stratified)
        ↓
   SMOTE (balance fraud:normal = 1:1)
        ↓
   Random Forest (100 estimators)
        ↓
   Evaluation (ROC AUC = 0.98)
        ↓
   Streamlit Deployment
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Ayush98sharma98/fraudshield-ai.git
cd fraudshield-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/` folder.

### 4. Train the model
```bash
python train.py
```

### 5. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser 🎉

---

## 📁 Project Structure

```
fraudshield-ai/
├── 📂 data/
│   └── creditcard.csv          # Dataset (download from Kaggle)
├── 📂 models/
│   ├── fraud_model.pkl          # Trained Random Forest model
│   └── scaler.pkl               # StandardScaler
├── 📂 notebooks/
│   └── fraud_detection.ipynb   # EDA & model training notebook
├── 📂 plots/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── class_distribution.png
├── 📄 app.py                    # Main Streamlit application
├── 📄 train.py                  # Model training script
├── 📄 setup.py                  # Cloud deployment setup
├── 📄 requirements.txt          # Python dependencies
├── 📄 packages.txt              # System dependencies
└── 📄 README.md
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle — ULB Machine Learning Group |
| Total Transactions | 284,807 |
| Fraud Cases | 492 (0.17%) |
| Normal Cases | 284,315 (99.83%) |
| Features | V1-V28 (PCA), Amount, Time |
| Time Period | September 2013, European cardholders |

> ⚠️ Dataset not included due to size (150MB). Download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place in `data/creditcard.csv`

---

## 🔑 Key Technical Decisions

### Why SMOTE?
The dataset had only 0.17% fraud cases. Without balancing, the model would simply predict "Normal" for everything and get 99.83% accuracy while catching zero fraud. SMOTE creates synthetic fraud examples to balance training data.

### Why Random Forest over Deep Learning?
For tabular data with engineered features, Random Forest performs comparably to deep learning with far less compute. It also provides feature importances for explainability — critical in banking for regulatory compliance.

### Why ROC AUC over Accuracy?
With imbalanced data, accuracy is misleading. ROC AUC measures the model's ability to distinguish between fraud and normal across all thresholds, making it the true performance metric here.

---

## 🎨 App Screenshots

### Dashboard
![Dashboard](plots/class_distribution.png)

### Model Performance
![Confusion Matrix](plots/confusion_matrix.png)

### Feature Importance
![Feature Importance](plots/feature_importance.png)

---

## 🔮 Future Improvements

- [ ] XGBoost / LightGBM integration
- [ ] SHAP explainability dashboard
- [ ] Real-time transaction streaming simulation
- [ ] Email alerts for fraud detection
- [ ] PostgreSQL database integration
- [ ] REST API with FastAPI
- [ ] Docker containerization

---

## 🎤 Interview Explanation

> *"I built a fraud detection system on 284K real transactions. The key challenge was class imbalance — only 0.17% were fraudulent. I used SMOTE to handle this, compared multiple models, and achieved 99.5% accuracy with 0.98 ROC AUC using Random Forest. I deployed it as a full Streamlit web app with real-time and batch prediction, 6 UI themes, and a downloadable results feature."*

---

## 📚 References

- Andrea Dal Pozzolo et al. *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE SSCI 2015
- Chawla, N.V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*
- [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 👨‍💻 Author

**Ayush Sharma**  
3rd Year CSE Student | Aspiring ML Engineer

[![GitHub](https://img.shields.io/badge/GitHub-Ayush98sharma98-black?style=flat&logo=github)](https://github.com/Ayush98sharma98)

## 🚀 Live Demo
👉 https://fraudshield-ai-severus.streamlit.app
```

### 3. Post on LinkedIn 🎯
```
🚀 Excited to share my latest project!

💳 FraudShield AI — Credit Card Fraud Detection System

✅ 99.5% Accuracy | 0.98 ROC AUC
✅ Real-time fraud detection
✅ Batch scanning for multiple transactions
✅ 6 UI themes

🔗 Live Demo: https://fraudshield-ai-severus.streamlit.app
🔗 GitHub: https://github.com/Ayush98sharma98/fraudshield-ai
🔗 linkdin: https://www.linkedin.com/feed/update/urn:li:activity:7441820691168989184/

#MachineLearning #DataScience #Python #MLEngineer


---

<div align="center">
⭐ Star this repo if you found it helpful!
</div>
