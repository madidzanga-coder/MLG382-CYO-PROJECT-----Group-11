import pandas as pd
import numpy as np
import os
import joblib
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

warnings.filterwarnings('ignore')
np.random.seed(42)

ARTIFACTS_DIR = 'artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def load_data(data_dir='data/processed'):
    """Load and prepare training and test data."""

    # Unscaled - for XGBoost (tree-based, scale invariant)
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_test  = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))

    # Scaled - for Logistic Regression and KNN (distance/gradient sensitive)
    X_train_scaled = pd.read_csv(os.path.join(data_dir, 'X_train_scaled.csv'))
    X_test_scaled  = pd.read_csv(os.path.join(data_dir, 'X_test_scaled.csv'))

    # Target labels
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))['stroke']
    y_test  = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))['stroke']

    # Convert boolean columns to int for sklearn compatibility
    bool_cols = X_train.select_dtypes(include='bool').columns.tolist()
    for df in [X_train, X_test, X_train_scaled, X_test_scaled]:
        df[bool_cols] = df[bool_cols].astype(int)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


def evaluate(model_name, y_test, y_pred):
    """Print evaluation metrics for a model."""
    print(f"\n{model_name} PERFORMANCE:")
    print(f"  Accuracy:   {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision:  {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall:     {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1-Score:   {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':

    # Load Data
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = load_data()
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape:     {X_test.shape}")

    # Model 1: Logistic Regression
    print("\n=== Training Logistic Regression ===")
    lr_grid = GridSearchCV(
        LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']},
        cv=3, scoring='f1', n_jobs=-1
    )
    lr_grid.fit(X_train_scaled, y_train)
    lr_best = lr_grid.best_estimator_
    print(f"Best params: {lr_grid.best_params_}")

    y_pred_lr = lr_best.predict(X_test_scaled)
    evaluate('Logistic Regression', y_test, y_pred_lr)

    joblib.dump(lr_best, os.path.join(ARTIFACTS_DIR, 'model_lr.pkl'))
    print("Logistic Regression model saved.")

    # Model 2: KNN 
    print("\n=== Training KNN ===")
    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        {'n_neighbors': [3, 5, 7, 11, 15], 'weights': ['uniform', 'distance']},
        cv=3, scoring='f1', n_jobs=-1
    )
    knn_grid.fit(X_train_scaled, y_train)
    knn_best = knn_grid.best_estimator_
    print(f"Best params: {knn_grid.best_params_}")

    y_pred_knn = knn_best.predict(X_test_scaled)
    evaluate('KNN', y_test, y_pred_knn)

    joblib.dump(knn_best, os.path.join(ARTIFACTS_DIR, 'model_knn.pkl'))
    print("KNN model saved.")

    # Model 3: XGBoost
    print("\n=== Training XGBoost ===")
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos

    xgb_grid = GridSearchCV(
        XGBClassifier(
            scale_pos_weight=scale, random_state=42,
            eval_metric='logloss', verbosity=0
        ),
        {'n_estimators': [50, 100, 200], 'max_depth': [4, 6, 8], 'learning_rate': [0.05, 0.1, 0.2]},
        cv=3, scoring='f1', n_jobs=-1
    )
    xgb_grid.fit(X_train, y_train)
    xgb_best = xgb_grid.best_estimator_
    print(f"Best params: {xgb_grid.best_params_}")

    y_pred_xgb = xgb_best.predict(X_test)
    evaluate('XGBoost', y_test, y_pred_xgb)

    joblib.dump(xgb_best, os.path.join(ARTIFACTS_DIR, 'model_xgb.pkl'))
    print("XGBoost model saved.")

    print("\nAll models trained and saved to artifacts/")