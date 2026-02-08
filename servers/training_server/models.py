"""
Machine Learning Models Module

This module provides real ML model implementations using scikit-learn.
Supports classification and regression tasks.
"""

import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

# Model registry
CLASSIFICATION_MODELS = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "svm": SVC,
    "decision_tree": DecisionTreeClassifier,
    "knn": KNeighborsClassifier,
    "naive_bayes": GaussianNB,
    "gradient_boosting": GradientBoostingClassifier,
    "mlp": MLPClassifier,
}

REGRESSION_MODELS = {
    "random_forest": RandomForestRegressor,
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "svr": SVR,
    "decision_tree": DecisionTreeRegressor,
    "knn": KNeighborsRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "mlp": MLPRegressor,
}


class MLModel:
    """Wrapper class for ML models."""
    
    def __init__(
        self,
        model_type: str,
        task_type: str = "classification",
        hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ML model.
        
        Args:
            model_type: Type of model (e.g., 'random_forest', 'logistic_regression')
            task_type: 'classification' or 'regression'
            hyperparameters: Model hyperparameters
        """
        self.model_type = model_type
        self.task_type = task_type
        self.hyperparameters = hyperparameters or {}
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.target_name = None
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on type and task."""
        if self.task_type == "classification":
            if self.model_type not in CLASSIFICATION_MODELS:
                raise ValueError(f"Unknown classification model: {self.model_type}")
            model_class = CLASSIFICATION_MODELS[self.model_type]
        elif self.task_type == "regression":
            if self.model_type not in REGRESSION_MODELS:
                raise ValueError(f"Unknown regression model: {self.model_type}")
            model_class = REGRESSION_MODELS[self.model_type]
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        
        # Filter valid hyperparameters for the model
        try:
            self.model = model_class(**self.hyperparameters)
            logger.info(f"Initialized {self.task_type} model: {self.model_type}")
        except TypeError as e:
            logger.warning(f"Invalid hyperparameters: {e}. Using defaults.")
            self.model = model_class()
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[list] = None,
        scale_features: bool = True,
        encode_labels: bool = False,
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: Names of features (optional)
            scale_features: Whether to scale features
            encode_labels: Whether to encode labels (for classification)
        
        Returns:
            Training metrics
        """
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        logger.info(f"Training {self.model_type} model...")
        
        # Scale features if requested
        if scale_features:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            logger.info("Features scaled")
        
        # Encode labels if requested (for classification with string labels)
        if encode_labels and self.task_type == "classification":
            self.label_encoder = LabelEncoder()
            y_train = self.label_encoder.fit_transform(y_train)
            logger.info(f"Labels encoded: {self.label_encoder.classes_}")
        
        # Train model
        self.model.fit(X_train, y_train)
        logger.info("Model training completed")
        
        # Calculate training metrics
        y_pred = self.model.predict(X_train)
        metrics = self._calculate_metrics(y_train, y_pred)
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict
        
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Scale features if scaler was used during training
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        predictions = self.model.predict(X)
        
        # Decode labels if encoder was used
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def validate(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Validate the model.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
        
        Returns:
            Validation metrics
        """
        logger.info("Validating model...")
        
        # Encode validation labels if encoder was used
        if self.label_encoder is not None:
            y_val = self.label_encoder.transform(y_val)
        
        y_pred = self.predict(X_val)
        
        # Re-encode predictions for metric calculation
        if self.label_encoder is not None:
            y_pred = self.label_encoder.transform(y_pred)
        
        metrics = self._calculate_metrics(y_val, y_pred, prefix="val_")
        logger.info(f"Validation completed: {metrics}")
        
        return metrics
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = "",
    ) -> Dict[str, Any]:
        """Calculate metrics based on task type."""
        metrics = {}
        
        if self.task_type == "classification":
            metrics[f"{prefix}accuracy"] = float(accuracy_score(y_true, y_pred))
            
            # Handle multi-class vs binary
            average = "binary" if len(np.unique(y_true)) == 2 else "weighted"
            
            metrics[f"{prefix}precision"] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
            metrics[f"{prefix}recall"] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
            metrics[f"{prefix}f1_score"] = float(f1_score(y_true, y_pred, average=average, zero_division=0))
            
        elif self.task_type == "regression":
            metrics[f"{prefix}mse"] = float(mean_squared_error(y_true, y_pred))
            metrics[f"{prefix}rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics[f"{prefix}mae"] = float(mean_absolute_error(y_true, y_pred))
            metrics[f"{prefix}r2_score"] = float(r2_score(y_true, y_pred))
        
        return metrics
    
    def save(self, path: str) -> str:
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
        
        Returns:
            Saved file path
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "model_type": self.model_type,
            "task_type": self.task_type,
            "hyperparameters": self.hyperparameters,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
        }
        
        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")
        
        return str(save_path)
    
    @classmethod
    def load(cls, path: str) -> "MLModel":
        """
        Load model from disk.
        
        Args:
            path: Path to the saved model
        
        Returns:
            Loaded MLModel instance
        """
        model_data = joblib.load(path)
        
        instance = cls(
            model_type=model_data["model_type"],
            task_type=model_data["task_type"],
            hyperparameters=model_data["hyperparameters"],
        )
        
        instance.model = model_data["model"]
        instance.scaler = model_data["scaler"]
        instance.label_encoder = model_data["label_encoder"]
        instance.feature_names = model_data.get("feature_names")
        instance.target_name = model_data.get("target_name")
        
        logger.info(f"Model loaded from {path}")
        
        return instance
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        if not hasattr(self.model, "feature_importances_"):
            return None
        
        if self.feature_names is None:
            return None
        
        importances = self.model.feature_importances_
        return {
            name: float(importance)
            for name, importance in zip(self.feature_names, importances)
        }


def load_dataset(
    file_path: str,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, str]:
    """
    Load dataset from file and split into train/test.
    
    Args:
        file_path: Path to dataset file (CSV)
        target_column: Name of target column
        test_size: Proportion of test set
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names, target_name
    """
    logger.info(f"Loading dataset from {file_path}")
    
    # Load data
    df = pd.read_csv(file_path)
    logger.info(f"Dataset shape: {df.shape}")
    
    # Separate features and target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    feature_names = X.columns.tolist()
    
    # Convert to numpy arrays
    X = X.values
    y = y.values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, feature_names, target_column
