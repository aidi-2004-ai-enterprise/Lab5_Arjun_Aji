"""
Lab 5 – Company Bankruptcy Prediction (16-Part End-to-End)
Covers:
1) Choosing Initial Models
2) Data Pre-processing
3) Handling Class Imbalance
4) Outlier Treatment
5) Avoiding Sampling Bias (PSI)
6) Data Normalization
7) Normality Testing
8) PCA (Dimensionality Reduction)
9) Feature Engineering Choices
10) Multicollinearity Check
11) Feature Selection Methods
12) Hyperparameter Tuning
13) Cross-Validation Strategy
14) Evaluation Metrics Selection
15) Monitoring Drift (PSI)
16) Model Explainability (SHAP / Importances)

Fast defaults; heavy steps are optional via flags below.
"""

# ======================
# Config / Fast Toggles
# ======================
DATA_PATH = "data.csv"         # <= change if needed
TARGET = "Bankrupt?"
RANDOM_STATE = 42

DO_OUTLIER_CAPPING = False     # Part 4: enable z-score winsorization
DO_VIF_PRUNE = False           # Part 10: iterative VIF pruning (requires statsmodels)
DO_PCA = False                 # Part 8: PCA for LR pipeline only
DO_SHAP = False                # Part 16: SHAP plots (set True if shap installed)
DO_GRID_REFINE = False         # Part 12: refine best model with GridSearch (slower)

# Hyperparam tuning sizes (keep modest = faster)
RANDOM_SEARCH_ITERS = 15
CV_FOLDS = 5

# ======================
# Imports
# ======================
import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, average_precision_score,
    f1_score, precision_recall_curve, roc_curve, brier_score_loss
)

from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

# ======================
# Utils
# ======================
def header(txt):
    print("\n" + "="*10 + " " + txt + " " + "="*10)

def zscore_cap(df, cols, z=4.0):
    cap = df.copy()
    for c in cols:
        s = pd.to_numeric(cap[c], errors='coerce')
        mu, sd = s.mean(), s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0: 
            continue
        cap[c] = s.clip(mu - z*sd, mu + z*sd)
    return cap

def psi(expected, actual, buckets=10):
    expected = pd.Series(expected).astype(float)
    actual = pd.Series(actual).astype(float)
    qs = np.linspace(0, 100, buckets+1)
    try:
        cuts = np.percentile(expected, qs)
    except Exception:
        return np.nan
    cuts[0] = -np.inf; cuts[-1] = np.inf
    e_bins = pd.cut(expected, cuts, include_lowest=True)
    a_bins = pd.cut(actual, cuts, include_lowest=True)
    e_pct = e_bins.value_counts(normalize=True, sort=False) + 1e-6
    a_pct = a_bins.value_counts(normalize=True, sort=False) + 1e-6
    return float(((a_pct - e_pct) * np.log(a_pct / e_pct)).sum())

def plot_cm(cm, title):
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    classes = ['Not Bankrupt','Bankrupt']
    ticks = np.arange(2)
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center')
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.tight_layout()
    plt.show()

def plot_roc_pr(y_true, y_prob, name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    # ROC
    fig = plt.figure()
    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC - {name}'); plt.legend()
    plt.show()

    # PR
    fig = plt.figure()
    plt.plot(rec, prec, label=f'{name} (AP={pr_auc:.3f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR - {name}'); plt.legend()
    plt.show()

# ======================
# PART 1: Choosing Initial Models
# ======================
header("PART 1: Choosing Initial Models")
print("- Logistic Regression (benchmark, interpretability)")
print("- Random Forest (nonlinear, robust, minimal preprocessing)")
print("- XGBoost (tabular SOTA); fallback to GradientBoosting if xgboost missing")

# ======================
# Load Data
# ======================
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates()
assert TARGET in df.columns, f"Target '{TARGET}' not found"
y = df[TARGET].astype(int)
X = df.drop(columns=[TARGET]).select_dtypes(include=[np.number]).copy()
print(f"Data shape: X={X.shape}, y={y.shape}, positive rate={y.mean():.4f}")

# ======================
# PART 2: Data Pre-processing
# ======================
header("PART 2: Data Pre-processing")
# Impute + scale for LR only; tree models use raw numeric inputs
num_cols = X.columns.tolist()
numeric_pre = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
# optional PCA for LR (part 8)
if DO_PCA:
    numeric_pre.steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))

# ======================
# PART 3: Handling Class Imbalance
# ======================
header("PART 3: Handling Class Imbalance")
print("- class_weight='balanced' for LR & RF")
print("- scale_pos_weight for XGB (ratio of negatives/positives)")
print("- stratified split used below")

# ======================
# Split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# ======================
# PART 4: Outlier Treatment (optional, conservative)
# ======================
header("PART 4: Outlier Treatment")
if DO_OUTLIER_CAPPING:
    X_train = zscore_cap(X_train, num_cols, z=4.0)
    X_test  = zscore_cap(X_test,  num_cols, z=4.0)
    print("Applied z-score capping (z=4).")
else:
    print("Skipped aggressive outlier removal (retain rare risk signals).")

# ======================
# PART 5 & 15: Sampling Bias + Drift (PSI between train/test)
# ======================
header("PART 5 & 15: PSI Drift Check (Train vs Test)")
psi_rows = []
for c in num_cols:
    try:
        psi_val = psi(X_train[c], X_test[c])
    except Exception:
        psi_val = np.nan
    psi_rows.append((c, psi_val))
psi_df = pd.DataFrame(psi_rows, columns=["feature","psi"]).sort_values("psi", ascending=False)
print(psi_df.head(10))
print("PSI guide: <0.1 none | 0.1–0.25 moderate | >0.25 significant shift")

# ======================
# PART 6: Data Normalization
# ======================
header("PART 6: Data Normalization")
print("- LR uses StandardScaler via pipeline")
print("- Tree/Boosting models are scale-invariant -> no scaling applied")

# ======================
# PART 7: Normality Testing (quick skewness scan)
# ======================
header("PART 7: Normality Testing")
skew = X_train.skew(numeric_only=True).sort_values(ascending=False)
print("Top 10 most skewed features:")
print(skew.head(10))
print("Note: no mandatory normality for trees; LR is robust with scaling.")

# ======================
# PART 8: PCA (skipped by default; toggle DO_PCA)
# ======================
header("PART 8: PCA Decision")
print(f"DO_PCA={DO_PCA}. Pros: reduce collinearity/overfit; Cons: lose interpretability.")

# ======================
# PART 9: Feature Engineering Choices
# ======================
header("PART 9: Feature Engineering Choices")
print("No new features added (domain-heavy ratios already present). Focus on selection.")

# ======================
# PART 10: Multicollinearity Check (Corr + optional VIF)
# ======================
header("PART 10: Multicollinearity Check")
corr = X_train.corr(numeric_only=True).abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_corr = [col for col in upper.columns if any(upper[col] >= 0.95)]
print(f"Highly correlated (|r|>=0.95) to drop: {len(drop_corr)}")
X_train_mc = X_train.drop(columns=drop_corr) if drop_corr else X_train.copy()
X_test_mc  = X_test.drop(columns=drop_corr) if drop_corr else X_test.copy()

if DO_VIF_PRUNE:
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        def vif_iter(df, thr=20.0, max_iter=10):
            kept = df.columns.tolist()
            Xv = df.values
            for _ in range(max_iter):
                vifs = [variance_inflation_factor(Xv, i) for i in range(Xv.shape[1])]
                v = pd.Series(vifs, index=kept)
                worst = v.idxmax(); vmax = v.max()
                if vmax > thr and len(kept) > 2:
                    kept.remove(worst)
                    Xv = df[kept].values
                else:
                    return kept, v
            return kept, v
        kept_cols, vser = vif_iter(X_train_mc, thr=20.0, max_iter=10)
        print(f"VIF kept: {len(kept_cols)} (dropped {X_train_mc.shape[1]-len(kept_cols)})")
        X_train_mc = X_train_mc[kept_cols]
        X_test_mc  = X_test_mc[kept_cols]
    except Exception as e:
        print("VIF step skipped:", e)

# ======================
# PART 11: Feature Selection (XGB importance + correlation)
# ======================
header("PART 11: Feature Selection")
# Train a quick booster (or gradient boosting fallback) to score features
if XGB_OK:
    pos_w = (y_train.value_counts()[0] / max(1, y_train.value_counts()[1]))
    fs_model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        scale_pos_weight=float(pos_w), eval_metric='logloss',
        tree_method='hist', random_state=RANDOM_STATE
    )
else:
    fs_model = GradientBoostingClassifier(random_state=RANDOM_STATE)

fs_model.fit(X_train_mc, y_train)
if hasattr(fs_model, "feature_importances_"):
    imp = pd.Series(fs_model.feature_importances_, index=X_train_mc.columns).sort_values(ascending=False)
else:
    # permutation importance fallback
    r = permutation_importance(fs_model, X_train_mc, y_train, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.Series(r.importances_mean, index=X_train_mc.columns).sort_values(ascending=False)

# keep features up to 95% cumulative importance (min 20)
cum = imp.cumsum() / imp.sum()
keep_feat = cum[cum <= 0.95].index.tolist()
if len(keep_feat) < min(20, len(imp)):
    keep_feat = imp.head(20).index.tolist()
print(f"Selected features: {len(keep_feat)} (cumulative importance ≤95% or top-20)")
X_train_fs = X_train_mc[keep_feat]; X_test_fs = X_test_mc[keep_feat]

# ======================
# PART 12: Hyperparameter Tuning (Random -> optional Grid refine)
# ======================
header("PART 12: Hyperparameter Tuning")
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# LR pipeline (impute+scale[+PCA]) on selected features
lr_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
] + ([("pca", PCA(n_components=0.95, svd_solver="full"))] if DO_PCA else []) + [
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE))
])

rf_model = RandomForestClassifier(
    n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
)

if XGB_OK:
    pos_w = (y_train.value_counts()[0] / max(1, y_train.value_counts()[1]))
    xgb_model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        scale_pos_weight=float(pos_w), eval_metric='logloss',
        tree_method='hist', random_state=RANDOM_STATE
    )
else:
    xgb_model = GradientBoostingClassifier(random_state=RANDOM_STATE)

models = {
    "LogisticRegression": lr_pipeline,
    "RandomForest": rf_model,
    "XGBoost" if XGB_OK else "GradientBoosting": xgb_model
}

search_spaces = {
    "LogisticRegression": {
        "clf__C": np.logspace(-2, 2, 10),
        "clf__solver": ["lbfgs", "liblinear"]
    },
    "RandomForest": {
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]
    },
    "XGBoost" if XGB_OK else "GradientBoosting": (
        {
            "n_estimators": [200, 400, 600],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [0.0, 1.0, 5.0],
        } if XGB_OK else {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1]
        }
    )
}

best = {}
for name, model in models.items():
    print(f"\nRandomizedSearchCV -> {name}")
    tuner = RandomizedSearchCV(
        estimator=model,
        param_distributions=search_spaces[name],
        n_iter=RANDOM_SEARCH_ITERS,
        scoring="average_precision",  # PR-AUC for imbalance
        cv=cv, n_jobs=-1, verbose=1, random_state=RANDOM_STATE
    )
    tuner.fit(X_train_fs, y_train)
    best[name] = tuner.best_estimator_
    print("Best params:", tuner.best_params_, "| Best PR-AUC:", f"{tuner.best_score_:.4f}")

# Optional grid refine on the single best-by-PR model (small grid)
if DO_GRID_REFINE:
    perf = []
    for n, m in best.items():
        # quick CV evaluation
        scores = []
        for tr, va in cv.split(X_train_fs, y_train):
            m.fit(X_train_fs.iloc[tr], y_train.iloc[tr])
            if hasattr(m, "predict_proba"):
                pr = average_precision_score(y_train.iloc[va], m.predict_proba(X_train_fs.iloc[va])[:,1])
            else:
                pr = average_precision_score(y_train.iloc[va], m.predict(X_train_fs.iloc[va]))
            scores.append(pr)
        perf.append((n, np.mean(scores)))
    perf.sort(key=lambda x: x[1], reverse=True)
    top_name = perf[0][0]
    print(f"\nGrid refine on top model: {top_name}")
    small_grid = {
        "LogisticRegression": {"clf__C": np.logspace(-1, 1, 5)},
        "RandomForest": {"n_estimators": [300, 400, 500], "max_depth": [None, 10]},
        "XGBoost": {"learning_rate": [0.03, 0.05, 0.08]} if XGB_OK else {"learning_rate": [0.05, 0.1]}
    }[top_name]
    grid = GridSearchCV(best[top_name], small_grid, scoring="average_precision", cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X_train_fs, y_train)
    best[top_name] = grid.best_estimator_
    print("Grid best params:", grid.best_params_, "| PR-AUC:", f"{grid.best_score_:.4f}")

# ======================
# PART 13 + 14: Cross-Validation strategy & Metrics (Test set eval)
# ======================
header("PART 13 & 14: Evaluation on Test Set (PR-AUC focus)")
summary = []
for name, model in best.items():
    model.fit(X_train_fs, y_train)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_fs)[:,1]
    else:
        # scale decision_function to [0,1] if needed
        preds = model.decision_function(X_test_fs)
        y_prob = (preds - preds.min()) / (preds.max()-preds.min()+1e-9)
    y_pred = (y_prob >= 0.5).astype(int)  # default threshold; can tune if desired

    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    pr  = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    print(f"\n{name} report:")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC-AUC={roc:.4f} | PR-AUC={pr:.4f} | F1={f1:.4f} | Brier={brier:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plot_cm(cm, f"Confusion Matrix - {name}")
    plot_roc_pr(y_test, y_prob, name)

    summary.append([name, roc, pr, f1, brier])

summary_df = pd.DataFrame(summary, columns=["Model","ROC-AUC","PR-AUC","F1","Brier"]).sort_values("PR-AUC", ascending=False)
print("\n=== Test Metrics (sorted by PR-AUC) ===")
print(summary_df.to_string(index=False))

# ======================
# PART 16: Model Explainability (SHAP or Importances)
# ======================
header("PART 16: Explainability")
best_row = summary_df.iloc[0]
best_name = best_row["Model"]
best_model = best[best_name]
print(f"Top model for explainability: {best_name}")

if DO_SHAP and SHAP_OK:
    try:
        # SHAP summary bar (fast)
        explainer = shap.Explainer(best_model, X_train_fs)
        shap_values = explainer(X_test_fs)
        plt.figure()
        shap.plots.bar(shap_values, show=False, max_display=15)
        plt.title(f"SHAP Summary (bar) - {best_name}")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("SHAP failed, falling back to importances:", e)

# Fallback: feature importances or permutation importance
if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=X_train_fs.columns).sort_values(ascending=False)
elif hasattr(best_model, "coef_"):
    coef = np.ravel(getattr(best_model, "coef_", np.zeros((1, X_train_fs.shape[1]))))
    imp = pd.Series(np.abs(coef), index=X_train_fs.columns).sort_values(ascending=False)
else:
    r = permutation_importance(best_model, X_test_fs, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.Series(r.importances_mean, index=X_train_fs.columns).sort_values(ascending=False)

print("\nTop 15 important features:")
print(imp.head(15))

plt.figure()
top = imp.head(15)
ypos = np.arange(len(top))
plt.barh(ypos, top.values)
plt.yticks(ypos, top.index)
plt.gca().invert_yaxis()
plt.title(f"Top 15 Feature Importances - {best_name}")
plt.tight_layout()
plt.show()

header("DONE")
print("All 16 parts executed with fast defaults. Toggle flags at top for deeper analysis (VIF, PCA, SHAP, Grid refine).")






