# -------------------------------
# Bankruptcy Prediction Full Visualization Script
# -------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
import shap
import os

# -------------------------------
# 1. Load Data
# -------------------------------
df = pd.read_csv("data.csv")
print(df.head())

target = "Bankrupt?"
X = df.drop(columns=[target])
y = df[target]

# Create folder to save plots
os.makedirs("plots", exist_ok=True)

# -------------------------------
# 2. Class Balance
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x=y)
total = len(y)
for p in plt.gca().patches:
    percentage = f'{100 * p.get_height()/total:.1f}%'
    plt.gca().annotate(percentage, (p.get_x()+p.get_width()/2., p.get_height()), 
                       ha='center', va='bottom')
plt.title("Class Distribution")
plt.savefig("plots/class_distribution.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 3. Feature Correlation Heatmap
# -------------------------------
plt.figure(figsize=(12,8))
corr = X.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap (Lower Triangle Only)")
plt.savefig("plots/correlation_heatmap.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 4. Feature Distributions (Histogram + KDE)
# -------------------------------
plt.figure(figsize=(20,15))
for i, col in enumerate(X.columns[:12]):
    plt.subplot(3,4,i+1)
    sns.histplot(X[col], kde=True, bins=30)
    plt.title(col)
plt.tight_layout()
plt.savefig("plots/feature_distributions.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 5. Outlier Detection
# -------------------------------
plt.figure(figsize=(15,8))
sns.boxplot(data=X.iloc[:, :10])
plt.xticks(rotation=90)
plt.title("Outlier Detection (First 10 Features)")
plt.savefig("plots/outliers_boxplot.png", bbox_inches="tight")
plt.show()

plt.figure(figsize=(15,8))
sns.violinplot(data=X.iloc[:, :10])
plt.xticks(rotation=90)
plt.title("Outlier Detection (Violin Plot, First 10 Features)")
plt.savefig("plots/outliers_violin.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 6. Train/Test Split & Scaling
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 7. Multicollinearity Check (VIF)
# -------------------------------
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)

# -------------------------------
# 8. Train Models
# -------------------------------
lr = LogisticRegression(class_weight="balanced", max_iter=1000)
rf = RandomForestClassifier(class_weight="balanced", random_state=42)
xgb = XGBClassifier(scale_pos_weight=len(y[y==0])/len(y[y==1]), eval_metric="logloss", use_label_encoder=False, random_state=42)

lr.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# -------------------------------
# 9. Random Forest Feature Importance
# -------------------------------
feat_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
feat_importances.head(20).plot(kind='bar')
plt.title("Random Forest Feature Importance (Top 20)")
plt.savefig("plots/rf_feature_importance.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 10. XGBoost Feature Importance
# -------------------------------
plt.figure(figsize=(10,6))
plot_importance(xgb, max_num_features=20)
plt.title("XGBoost Feature Importance (Top 20)")
plt.savefig("plots/xgb_feature_importance.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 11. Confusion Matrices
# -------------------------------
models = {"Logistic Regression": lr, "Random Forest": rf, "XGBoost": xgb}
for name, model in models.items():
    if name == "Logistic Regression":
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6,4))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f"Normalized Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"plots/confusion_matrix_{name.replace(' ','_')}.png", bbox_inches="tight")
    plt.show()

# -------------------------------
# 12. ROC Curves
# -------------------------------
plt.figure(figsize=(8,6))
for name, model in models.items():
    if name == "Logistic Regression":
        y_prob = model.predict_proba(X_test_scaled)[:,1]
    else:
        y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
plt.plot([0,1],[0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.savefig("plots/roc_curves.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 13. Precision-Recall Curves
# -------------------------------
plt.figure(figsize=(8,6))
for name, model in models.items():
    if name == "Logistic Regression":
        y_prob = model.predict_proba(X_test_scaled)[:,1]
    else:
        y_prob = model.predict_proba(X_test)[:,1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, label=name)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.savefig("plots/precision_recall_curves.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 14. SHAP Summary Plot (XGBoost)
# -------------------------------
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, max_display=20)

# Optional: force plot for first prediction
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])

# -------------------------------
# 15. Population Stability Index (PSI)
# -------------------------------
def calculate_psi(expected, actual, buckets=10):
    def scale_range (input, min_val, max_val):
        input_std = (input - input.min()) / (input.max() - input.min())
        return input_std * (max_val - min_val) + min_val

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = scale_range(np.arange(0, buckets + 1), min_val=min(expected.min(), actual.min()), max_val=max(expected.max(), actual.max()))

    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    psi_value = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    return psi_value

psi_scores = {col: calculate_psi(X_train[col], X_test[col]) for col in X.columns}
psi_df = pd.DataFrame(list(psi_scores.items()), columns=["Feature", "PSI"])
plt.figure(figsize=(10,6))
sns.barplot(x="PSI", y="Feature", data=psi_df.sort_values("PSI", ascending=False))
plt.title("Population Stability Index by Feature")
plt.savefig("plots/psi.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 16. Final Evaluation Report
# -------------------------------
for name, model in models.items():
    if name == "Logistic Regression":
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred))
