# Bankruptcy Prediction – Lab 5

## 📌 Overview
This project implements a **Bankruptcy Prediction System** using three machine learning models:
- Logistic Regression
- Random Forest
- XGBoost

The workflow follows **16 clearly defined steps**, from data preprocessing to explainability.  
The goal is to predict whether a company will go bankrupt based on financial ratios.

---

## 📂 Project Structure
```bash
│── data/
│ ├── data.csv
│
│── lab5_analysis.ipynb
│
│── test.py
│
│── README.md
│── requirements.txt

```

---

## 🛠 16-Part Approach Summary

1. **Model Selection** – Logistic Regression, Random Forest, XGBoost  
2. **Data Preprocessing** – Cleaning, removing duplicates, irrelevant features  
3. **Class Imbalance Handling** – Class weights, stratified splits  
4. **Outlier Handling** – Kept extreme values for interpretability, scaled for LR  
5. **Sampling Bias Check** – Population Stability Index (PSI)  
6. **Normalization** – StandardScaler for LR only  
7. **Normality Testing** – Addressed highly skewed features if needed  
8. **PCA** – Skipped to preserve interpretability  
9. **Feature Engineering** – Minimal, avoided overfitting  
10. **Multicollinearity** – Removed features with correlation > 0.9  
11. **Feature Selection** – Used XGBoost importance + correlation checks  
12. **Hyperparameter Tuning** – Random Search + Grid Search  
13. **Cross-Validation** – Stratified K-Fold  
14. **Evaluation Metrics** – F1, ROC-AUC, PR-AUC  
15. **Model Drift Monitoring** – PSI for post-deployment checks  
16. **Explainability** – SHAP values for model interpretation  

---

## 📊 Visualizations
The following plots are generated:
- Correlation heatmap
- ROC curves for all models
- Precision-Recall curves
- Feature importance plots
- SHAP summary plot

---

## ▶ How to Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
2️⃣ Run Analysis
```bash
python src/modeling.py
Or open the Jupyter Notebook:
```
```bash
jupyter notebook notebooks/lab5_analysis.ipynb
```
3️⃣ View Results
Evaluation metrics will be printed in the console.

Plots will be saved in outputs/.

SHAP explanations will be in outputs/shap_plots/.

📦 Requirements
```bash
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- shap
```
📈 Results Summary
- Best Model: XGBoost (highest ROC-AUC and F1 score)

- Logistic Regression offered better interpretability

- Random Forest performed close to XGBoost but with slightly lower precision

📜 License
This is an academic project for Lab 5.
Feel free to adapt for learning purposes.

