import os
import warnings
import logging
import numpy as np
import joblib
import xgboost as xgb
from dateutil.relativedelta import relativedelta
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix

from utils import (
    load_and_preprocess_data, 
    feature_engineering_common, 
    create_directories, 
    get_feature_columns, 
    MODELS_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_COL = 'Target_Crisis'
XGB_DIR = os.path.join(MODELS_DIR, 'XGBoost')


def _calculate_class_weight(y_train: np.ndarray) -> float:
    """Calculate scale_pos_weight for imbalanced classification."""
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    return num_neg / num_pos if num_pos > 0 else 1.0


def _log_model_metrics(y_test: np.ndarray, probs: np.ndarray, preds: np.ndarray) -> None:
    """Log model evaluation metrics."""
    unique_test = np.unique(y_test)
    
    # AUC Score
    if len(unique_test) > 1:
        auc = roc_auc_score(y_test, probs)
        logger.info("TEST AUC: %.4f", auc)
    else:
        logger.info("TEST AUC: N/A (Single class)")
    
    # Precision and Recall
    rec = recall_score(y_test, preds, zero_division=0)
    prec = precision_score(y_test, preds, zero_division=0)
    logger.info("Recall: %.4f | Precision: %.4f", rec, prec)

    # Confusion Matrix
    if len(unique_test) > 1 or (len(unique_test) == 1 and unique_test[0] in preds):
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        logger.info("CONFUSION TN: %d | FP: %d | FN: %d | TP: %d", tn, fp, fn, tp)


def train_and_evaluate_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> None:
    """
    Train, evaluate, and save a single XGBoost model.
    
    Args:
        model_name: Name for saving the model.
        X_train, y_train: Training data and labels.
        X_val, y_val: Validation data and labels.
        X_test, y_test: Test data and labels.
    """
    logger.info("[%s] Training...", model_name)
    logger.info("Train: %d | Val: %d | Test: %d", len(X_train), len(X_val), len(X_test))

    num_pos = y_train.sum()
    if num_pos == 0:
        logger.error("No crisis samples in training set. Skipping.")
        return

    scale_pos_weight = _calculate_class_weight(y_train)
    logger.info("Class ratio -> Normal:%d, Crisis:%d (1:%.2f)", len(y_train)-num_pos, num_pos, scale_pos_weight)

    # Determine eval metric based on validation set
    eval_metric = 'auc' if len(np.unique(y_val)) >= 2 else 'logloss'
    eval_set = [(X_train, y_train), (X_val, y_val)]

    model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        eval_metric=eval_metric,
        early_stopping_rounds=100,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    logger.info("Best iteration: %d", model.best_iteration)

    # Evaluate on test set
    if len(X_test) > 0:
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        _log_model_metrics(y_test, probs, preds)
    else:
        logger.warning("Empty test set!")

    # Save model
    model_path = os.path.join(XGB_DIR, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    logger.info("SAVED: %s", model_path)


def _create_period_masks(df, start, train_end, val_end, test_end, include_end=False):
    """Create boolean masks for train/val/test periods."""
    train_mask = (df['Date'] >= start) & (df['Date'] < train_end)
    val_mask = (df['Date'] >= train_end) & (df['Date'] < val_end)
    
    if include_end:
        test_mask = (df['Date'] >= val_end) & (df['Date'] <= test_end)
    else:
        test_mask = (df['Date'] >= val_end) & (df['Date'] < test_end)
    
    return train_mask, val_mask, test_mask


def train_xgboost_dual_regime() -> None:
    """Train dual XGBoost models for early and late market periods."""
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    create_directories()
    
    logger.info("Loading data")
    raw_df = load_and_preprocess_data()
    
    logger.info("Feature engineering")
    df = feature_engineering_common(raw_df)
    
    feature_cols = get_feature_columns(df)
    logger.info("Feature count: %d", len(feature_cols))
    
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    logger.info("Date range: %s - %s", start_date.date(), end_date.date())

    # Model 1: Early Period
    m1_train_end = start_date + relativedelta(years=5)
    m1_val_end = m1_train_end + relativedelta(months=6)
    m1_test_end = m1_val_end + relativedelta(months=6)

    logger.info("MODEL 1: EARLY PERIOD")
    logger.info("Train: %s -> %s", start_date.date(), m1_train_end.date())
    logger.info("Val: %s -> %s", m1_train_end.date(), m1_val_end.date())
    logger.info("Test: %s -> %s", m1_val_end.date(), m1_test_end.date())

    mask_train1, mask_val1, mask_test1 = _create_period_masks(
        df, start_date, m1_train_end, m1_val_end, m1_test_end
    )

    train_and_evaluate_model(
        "xgb_model_early",
        df.loc[mask_train1, feature_cols], df.loc[mask_train1, TARGET_COL],
        df.loc[mask_val1, feature_cols], df.loc[mask_val1, TARGET_COL],
        df.loc[mask_test1, feature_cols], df.loc[mask_test1, TARGET_COL]
    )

    # Model 2: Late Period
    m2_test_start = end_date - relativedelta(months=3)
    m2_val_start = m2_test_start - relativedelta(months=3)
    m2_train_start = m2_val_start - relativedelta(years=5)

    logger.info("MODEL 2: LATE PERIOD")
    logger.info("Train: %s -> %s", m2_train_start.date(), m2_val_start.date())
    logger.info("Val: %s -> %s", m2_val_start.date(), m2_test_start.date())
    logger.info("Test: %s -> %s", m2_test_start.date(), end_date.date())

    mask_train2, mask_val2, mask_test2 = _create_period_masks(
        df, m2_train_start, m2_val_start, m2_test_start, end_date, include_end=True
    )

    train_and_evaluate_model(
        "xgb_model_late",
        df.loc[mask_train2, feature_cols], df.loc[mask_train2, TARGET_COL],
        df.loc[mask_val2, feature_cols], df.loc[mask_val2, TARGET_COL],
        df.loc[mask_test2, feature_cols], df.loc[mask_test2, TARGET_COL]
    )

    logger.info("Dual model training completed.")


if __name__ == "__main__":
    train_xgboost_dual_regime()