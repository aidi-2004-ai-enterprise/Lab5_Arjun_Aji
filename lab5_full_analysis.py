#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lab 5 – Full Binary Classification Analysis
Dataset: Company Bankruptcy Prediction (assumed CSV at ./data.csv with target column 'Bankrupt?')

What this script does (end-to-end):
1) Loads data and performs quick EDA summaries
2) Splits train/test with stratification
3) Optional outlier capping (disabled by default)
4) Multicollinearity checks (VIF) with optional iterative pruning
5) Feature selection options (correlation filter + model-based importance)
6) Preprocessing per model (scaling for LR; none for trees/boosting)
7) Class imbalance handling (class_weight or scale_pos_weight)
8) Hyperparameter tuning via RandomizedSearchCV
9) Cross-validation with StratifiedKFold
10) Evaluation: F1, ROC-AUC, PR-AUC, Brier, confusion matrix, PR & ROC curves
11) PSI drift check (train vs test) for every feature
12) Explainability: SHAP (if installed) else permutation importance / model importances
13) Persistence: save best model (and scaler if needed) to .pkl
14) Exports: metrics.csv, psi.csv, feature_importances.csv, and plots/*.png

USAGE:
    python lab5_full_analysis.py --data /path/to/data.csv --target "Bankrupt?"

Notes:
- Uses only matplotlib (no seaborn) for plots.
- If xgboost or shap are not installed, the script falls back gracefully.
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, precision_recall_curve, roc_curve, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer

from scipy import stats
from math import isfinite

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

import joblib
import matplotlib.pyplot as plt

# --------------------------- Utility Functions ---------------------------

def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def summarize_dataframe(df, target):
    print("=== Shape ===")
    print(df.shape)
    print("\n=== Columns ===")
    print(list(df.columns))
    print("\n=== Missing by column (top 10) ===")
    print(df.isna().sum().sort_values(ascending=False).head(10))
    print("\n=== Target distribution ===")
    print(df[target].value_counts(dropna=False))

def zscore_cap(df, cols, z=4.0):
    """Winsorize extreme outliers by z-score threshold (two-sided)."""
    capped = df.copy()
    for c in cols:
        s = capped[c].astype(float)
        mu, sd = s.mean(), s.std(ddof=0)
        if sd == 0 or not np.isfinite(sd):
            continue
        upper = mu + z*sd
        lower = mu - z*sd
        s = s.clip(lower=lower, upper=upper)
        capped[c] = s
    return capped

def calculate_psi(expected, actual, buckets=10):
    """Population Stability Index between two numeric arrays."""
    expected = pd.Series(expected).astype(float)
    actual = pd.Series(actual).astype(float)
    # rank-based qcut for robust binning
    try:
        expected_bins = pd.qcut(expected.rank(method='first'), buckets, duplicates='drop')
        actual_bins = pd.qcut(actual.rank(method='first'), buckets, duplicates='drop')
    except Exception:
        # if not enough unique values, fall back to cut
        expected_bins = pd.cut(expected, buckets, duplicates='drop')
        actual_bins = pd.cut(actual, buckets, duplicates='drop')
    exp_pct = expected_bins.value_counts(normalize=True, sort=False)
    act_pct = actual_bins.value_counts(normalize=True, sort=False)
    # align indexes
    all_idx = sorted(set(exp_pct.index).union(set(act_pct.index)), key=lambda x: str(x))
    exp_pct = exp_pct.reindex(all_idx).fillna(0) + 1e-6
    act_pct = act_pct.reindex(all_idx).fillna(0) + 1e-6
    psi_vals = (exp_pct - act_pct) * np.log(exp_pct / act_pct)
    return float(psi_vals.sum())

def vif_scores(X_df, vif_threshold=20.0, max_iter=10):
    """Compute VIF iteratively; optionally drop high-VIF features."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    cols = list(X_df.columns)
    X_curr = X_df.copy()
    dropped = []
    for _ in range(max_iter):
        vifs = []
        for i in range(X_curr.shape[1]):
            try:
                vifs.append(variance_inflation_factor(X_curr.values, i))
            except Exception:
                vifs.append(np.nan)
        vif_series = pd.Series(vifs, index=X_curr.columns)
        worst = vif_series.idxmax()
        worst_val = vif_series.max()
        if isfinite(worst_val) and worst_val > vif_threshold and len(cols) > 2:
            dropped.append((worst, worst_val))
            X_curr = X_curr.drop(columns=[worst])
            cols.remove(worst)
        else:
            break
    return pd.Series(vifs, index=X_df.columns), dropped, X_curr.columns.tolist()

def correlation_filter(X_df, threshold=0.95):
    """Drop one of each highly correlated pair (|r|>=threshold)."""
    corr = X_df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [column for column in upper.columns if any(upper[column] >= threshold)]
    keep_cols = [c for c in X_df.columns if c not in drop_cols]
    return keep_cols, drop_cols

def plot_confusion_matrix(cm, classes, outpath):
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)

def plot_roc_curve(y_true, y_prob, outpath):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig = plt.figure()
    plt.plot(fpr, tpr, label='ROC')
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)

def plot_pr_curve(y_true, y_prob, outpath):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig = plt.figure()
    plt.plot(recall, precision, label='PR')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)

# --------------------------- Main ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data.csv', help='Path to CSV')
    parser.add_argument('--target', type=str, default='Bankrupt?', help='Target column name')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--do_outlier_cap', action='store_true')
    parser.add_argument('--do_vif_prune', action='store_true')
    parser.add_argument('--do_corr_filter', action='store_true')
    parser.add_argument('--pca', action='store_true', help='(Optional) if you want to add PCA later')
    args = parser.parse_args()

    safe_mkdir('plots')
    out_metrics_csv = 'metrics.csv'
    out_importance_csv = 'feature_importances.csv'
    out_psi_csv = 'psi.csv'
    model_path = 'best_model.pkl'
    scaler_path = 'scaler.pkl'

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset.")
    summarize_dataframe(df, args.target)

    # Select numeric columns (dataset is numeric by design)
    y = df[args.target].astype(int)
    X = df.drop(columns=[args.target]).select_dtypes(include=[np.number]).copy()

    # Optional outlier capping
    if args.do_outlier_cap:
        X = zscore_cap(X, X.columns, z=4.0)

    # Optional correlation filter
    if args.do_corr_filter:
        keep_cols, dropped_cols = correlation_filter(X, threshold=0.95)
        print(f"Correlation filter dropped {len(dropped_cols)} columns.")
        X = X[keep_cols]

    # Optional VIF prune (requires statsmodels)
    if args.do_vif_prune:
        try:
            vifs, dropped, keep_cols = vif_scores(X, vif_threshold=20.0, max_iter=10)
            print("Final kept columns after VIF prune:", len(keep_cols))
            if dropped:
                print("Dropped for high VIF:", dropped)
            X = X[keep_cols]
        except Exception as e:
            print("VIF prune skipped due to error:", e)

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    # PSI between train and test
    psi_records = []
    for col in X.columns:
        try:
            psi_val = calculate_psi(X_train[col], X_test[col], buckets=10)
            psi_records.append({'feature': col, 'psi': psi_val})
        except Exception as e:
            psi_records.append({'feature': col, 'psi': np.nan})
    psi_df = pd.DataFrame(psi_records).sort_values('psi', ascending=False)
    psi_df.to_csv(out_psi_csv, index=False)
    print("\nTop PSI features:\n", psi_df.head(10))

    # Define models
    lr_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=500, n_jobs=None, random_state=args.random_state))
    ])

    rf_clf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_split=2, min_samples_leaf=1,
        class_weight='balanced', random_state=args.random_state, n_jobs=-1
    )

    if XGB_AVAILABLE:
        pos_weight = (y_train.value_counts()[0] / max(1, y_train.value_counts()[1]))
        xgb_clf = XGBClassifier(
            n_estimators=600, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            scale_pos_weight=float(pos_weight), eval_metric='logloss',
            tree_method='hist', random_state=args.random_state
        )
    else:
        xgb_clf = GradientBoostingClassifier(random_state=args.random_state)  # fallback

    models = {
        'LogisticRegression': lr_pipeline,
        'RandomForest': rf_clf,
        'XGBoost' if XGB_AVAILABLE else 'GradientBoosting': xgb_clf
    }

    # Hyperparameter tuning (RandomizedSearchCV)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)

    # LR params (mainly C and penalty)
    lr_space = {
        'clf__C': np.logspace(-2, 2, 20),
        'clf__penalty': ['l2'],
        'clf__solver': ['lbfgs', 'liblinear']
    }

    rf_space = {
        'n_estimators': [200, 400, 600, 800],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    if XGB_AVAILABLE:
        xgb_space = {
            'n_estimators': [300, 600, 900],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.03, 0.05, 0.1],
            'subsample': [0.7, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.9, 1.0],
            'reg_lambda': [0.0, 1.0, 5.0],
            # scale_pos_weight is set from class ratio; keep fixed
        }
    else:
        xgb_space = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }

    tuned_models = {}
    for name, model in models.items():
        if name == 'LogisticRegression':
            param_distributions = lr_space
            scoring = 'average_precision'  # PR-AUC focus due to imbalance
        elif name == 'RandomForest':
            param_distributions = rf_space
            scoring = 'average_precision'
        else:
            param_distributions = xgb_space
            scoring = 'average_precision'

        print(f"\n=== Tuning {name} ===")
        tuner = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=20,
            scoring=scoring,
            cv=cv,
            verbose=1,
            n_jobs=-1,
            random_state=args.random_state
        )
        tuner.fit(X_train, y_train)
        tuned_models[name] = tuner.best_estimator_
        print(f"Best params for {name}: {tuner.best_params_} | Best {scoring}: {tuner.best_score_:.4f}")

    # Evaluate tuned models
    rows = []
    best_name, best_score = None, -np.inf
    prob_cache = {}

    for name, model in tuned_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            # decision_function fallback
            if hasattr(model, "decision_function"):
                scores = model.decision_function(X_test)
                # scale scores to [0,1] via min-max for curves
                y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
            else:
                y_prob = y_pred.astype(float)

        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        prauc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)

        rows.append({
            'model': name, 'f1': f1, 'roc_auc': roc, 'pr_auc': prauc, 'brier': brier
        })
        prob_cache[name] = (y_pred, y_prob)

        # track best by PR-AUC
        if prauc > best_score:
            best_score = prauc
            best_name = name

        # Confusion Matrix & Curves
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, classes=['No', 'Yes'], outpath=f'plots/{name}_confusion_matrix.png')
        plot_roc_curve(y_test, y_prob, outpath=f'plots/{name}_roc.png')
        plot_pr_curve(y_test, y_prob, outpath=f'plots/{name}_pr.png')

        print(f"\n{name} report:\n", classification_report(y_test, y_pred, digits=4))

    metrics_df = pd.DataFrame(rows).sort_values('pr_auc', ascending=False)
    metrics_df.to_csv(out_metrics_csv, index=False)
    print("\n=== Test Metrics (sorted by PR-AUC) ===")
    print(metrics_df)

    # Save best model (and scaler if LR pipeline) 
    best_model = tuned_models[best_name]
    joblib.dump(best_model, model_path)
    print(f"\nSaved best model: {best_name} -> {model_path}")
    if best_name == 'LogisticRegression':
        # scaler is inside pipeline; saving whole pipeline is enough
        pass

    # Feature Importance
    feat_imp = []
    if best_name in ['RandomForest', 'XGBoost', 'GradientBoosting'] and hasattr(best_model, 'feature_importances_'):
        imps = best_model.feature_importances_
        feat_imp = pd.DataFrame({'feature': X.columns, 'importance': imps}).sort_values('importance', ascending=False)
    else:
        # permutation importance (model-agnostic)
        print("\nComputing permutation importance (this may take a moment)...")
        r = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=args.random_state, n_jobs=-1)
        feat_imp = pd.DataFrame({'feature': X.columns, 'importance': r.importances_mean}).sort_values('importance', ascending=False)
    feat_imp.to_csv(out_importance_csv, index=False)

    # Plot top-15 importances
    if len(feat_imp) > 0:
        top = feat_imp.head(15)
        fig = plt.figure()
        y_pos = np.arange(len(top))
        plt.barh(y_pos, top['importance'].values)
        plt.yticks(y_pos, top['feature'].values)
        plt.gca().invert_yaxis()
        plt.title(f'Top 15 Feature Importances ({best_name})')
        fig.savefig('plots/feature_importances_top15.png', bbox_inches='tight')
        plt.close(fig)

    # SHAP Explainability (if available and model supports it)
    if SHAP_AVAILABLE:
        try:
            explainer = None
            if best_name in ['RandomForest', 'GradientBoosting']:
                # use TreeExplainer if supported; else KernelExplainer (slow)
                explainer = shap.Explainer(best_model, X_train)
            elif best_name == 'LogisticRegression':
                explainer = shap.Explainer(best_model, X_train)
            elif best_name == 'XGBoost':
                explainer = shap.Explainer(best_model, X_train, feature_names=X.columns.tolist())
            if explainer is not None:
                shap_vals = explainer(X_test)
                # summary plot (bar)
                fig = plt.figure()
                shap.plots.bar(shap_vals, show=False, max_display=15)
                fig = plt.gcf()
                fig.savefig('plots/shap_summary_bar.png', bbox_inches='tight')
                plt.close(fig)
        except Exception as e:
            print("SHAP explanation skipped due to error:", e)

    print("\nAll artifacts saved:")
    print(f" - {out_metrics_csv}")
    print(f" - {out_psi_csv}")
    print(f" - {out_importance_csv}")
    print(" - plots/*.png")
    print(f" - {model_path}")


if __name__ == '__main__':
    main()
