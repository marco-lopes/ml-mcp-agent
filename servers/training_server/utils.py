"""
Utility functions for data processing and model evaluation.
"""

import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, f1_score, mean_squared_error

logger = logging.getLogger(__name__)


def validate_dataset(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    Validate dataset and return information.
    
    Args:
        df: DataFrame to validate
        target_column: Name of target column
    
    Returns:
        Dictionary with dataset information
    """
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "target_column": target_column,
    }
    
    if target_column in df.columns:
        target_values = df[target_column].value_counts()
        info["target_distribution"] = target_values.to_dict()
        info["num_classes"] = len(target_values)
    
    return info


def preprocess_features(
    df: pd.DataFrame,
    target_column: str,
    handle_missing: str = "drop",
    encode_categorical: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess features in a dataset.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        handle_missing: How to handle missing values ('drop', 'mean', 'median')
        encode_categorical: Whether to encode categorical variables
    
    Returns:
        Processed features and target
    """
    df = df.copy()
    
    # Separate features and target
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Handle missing values
    if handle_missing == "drop":
        # Drop rows with missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
    elif handle_missing == "mean":
        # Fill numeric columns with mean
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
    elif handle_missing == "median":
        # Fill numeric columns with median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    # Encode categorical variables
    if encode_categorical:
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            X[col] = pd.Categorical(X[col]).codes
    
    logger.info(f"Preprocessed dataset: {X.shape}")
    
    return X, y


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    task_type: str = "classification",
) -> Dict[str, Any]:
    """
    Perform cross-validation on a model.
    
    Args:
        model: Scikit-learn model
        X: Features
        y: Target
        cv: Number of folds
        task_type: 'classification' or 'regression'
    
    Returns:
        Cross-validation results
    """
    logger.info(f"Performing {cv}-fold cross-validation...")
    
    if task_type == "classification":
        scoring = {
            "accuracy": make_scorer(accuracy_score),
            "f1": make_scorer(f1_score, average="weighted", zero_division=0),
        }
    else:
        scoring = {
            "mse": make_scorer(mean_squared_error, greater_is_better=False),
            "rmse": make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False),
        }
    
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
    )
    
    # Calculate mean and std for each metric
    results = {}
    for key, values in cv_results.items():
        if key.startswith("test_") or key.startswith("train_"):
            metric_name = key.replace("test_", "cv_").replace("train_", "train_cv_")
            results[f"{metric_name}_mean"] = float(np.mean(values))
            results[f"{metric_name}_std"] = float(np.std(values))
    
    logger.info(f"Cross-validation completed: {results}")
    
    return results


def get_model_summary(model, feature_names: List[str] = None) -> Dict[str, Any]:
    """
    Get a summary of the model.
    
    Args:
        model: Trained scikit-learn model
        feature_names: List of feature names
    
    Returns:
        Model summary
    """
    summary = {
        "model_type": type(model).__name__,
        "parameters": model.get_params(),
    }
    
    # Add feature importance if available
    if hasattr(model, "feature_importances_") and feature_names:
        importances = model.feature_importances_
        feature_importance = {
            name: float(imp)
            for name, imp in sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True,
            )
        }
        summary["feature_importance"] = feature_importance
    
    # Add coefficients if available (linear models)
    if hasattr(model, "coef_") and feature_names:
        if len(model.coef_.shape) == 1:
            # Binary classification or regression
            coefficients = {
                name: float(coef)
                for name, coef in zip(feature_names, model.coef_)
            }
        else:
            # Multi-class classification
            coefficients = {
                f"class_{i}": {
                    name: float(coef)
                    for name, coef in zip(feature_names, model.coef_[i])
                }
                for i in range(model.coef_.shape[0])
            }
        summary["coefficients"] = coefficients
    
    return summary


def generate_dataset_report(
    df: pd.DataFrame,
    target_column: str,
) -> str:
    """
    Generate a text report about the dataset.
    
    Args:
        df: DataFrame
        target_column: Name of target column
    
    Returns:
        Text report
    """
    report = []
    report.append("=" * 60)
    report.append("DATASET REPORT")
    report.append("=" * 60)
    report.append(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")
    report.append(f"Target column: {target_column}")
    
    # Feature types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    report.append(f"\nNumeric features: {len(numeric_cols)}")
    report.append(f"Categorical features: {len(categorical_cols)}")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        report.append("\nMissing values:")
        for col, count in missing[missing > 0].items():
            report.append(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    else:
        report.append("\nNo missing values")
    
    # Target distribution
    if target_column in df.columns:
        report.append(f"\nTarget distribution:")
        value_counts = df[target_column].value_counts()
        for value, count in value_counts.items():
            report.append(f"  {value}: {count} ({count/len(df)*100:.1f}%)")
    
    # Basic statistics for numeric columns
    if len(numeric_cols) > 0:
        report.append("\nNumeric features statistics:")
        stats = df[numeric_cols].describe()
        report.append(str(stats))
    
    report.append("=" * 60)
    
    return "\n".join(report)
