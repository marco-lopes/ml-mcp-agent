#!/usr/bin/env python3
"""
MCP Server for Machine Learning Model Training

This server provides tools for training, validating, and managing ML models.
Uses scikit-learn for real model training.
It communicates with clients via the Model Context Protocol using JSON-RPC.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from mcp.server.fastmcp import FastMCP

# Add current directory to path to import models module
sys.path.insert(0, str(Path(__file__).parent))

from models import MLModel, load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("ml-training-server")

# Storage for model metadata
MODELS_DIR = Path("/tmp/ml_models")
MODELS_DIR.mkdir(exist_ok=True)

# In-memory storage for model metadata and instances
models_registry: dict[str, dict[str, Any]] = {}
active_models: dict[str, MLModel] = {}
MAX_ACTIVE_MODELS = 50


def _register_active_model(model_id: str, ml_model: MLModel) -> None:
    """Register a model in active_models, evicting oldest if at capacity."""
    if len(active_models) >= MAX_ACTIVE_MODELS:
        oldest_key = next(iter(active_models))
        del active_models[oldest_key]
        logger.info(f"Evicted oldest model from cache: {oldest_key}")
    active_models[model_id] = ml_model


def save_model_metadata(model_id: str, metadata: dict[str, Any]) -> None:
    """Save model metadata to registry."""
    models_registry[model_id] = metadata
    metadata_file = MODELS_DIR / f"{model_id}_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Model metadata saved: {model_id}")


def load_model_metadata(model_id: str) -> Optional[dict[str, Any]]:
    """Load model metadata from registry."""
    if model_id in models_registry:
        return models_registry[model_id]
    
    metadata_file = MODELS_DIR / f"{model_id}_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            return json.load(f)
    return None


@mcp.tool()
async def train_model(
    model_name: str,
    model_type: str,
    dataset_path: str,
    target_column: str,
    task_type: str = "classification",
    hyperparameters: Optional[dict[str, Any]] = None,
    test_size: float = 0.2,
    scale_features: bool = True,
    random_state: int = 42,
) -> str:
    """
    Train a machine learning model using scikit-learn.
    
    Args:
        model_name: Unique name for the model
        model_type: Type of model (random_forest, logistic_regression, svm, etc.)
        dataset_path: Path to the dataset CSV file
        target_column: Name of the target column in the dataset
        task_type: 'classification' or 'regression'
        hyperparameters: Model hyperparameters (optional)
        test_size: Proportion of data for testing (default: 0.2)
        scale_features: Whether to scale features (default: True)
        random_state: Random seed for reproducibility
    
    Returns:
        JSON string with training results
    """
    try:
        logger.info(f"Training model: {model_name} ({model_type})")

        # Validate dataset path
        if not Path(dataset_path).is_file():
            return json.dumps({
                "success": False,
                "error": f"Dataset file not found: {dataset_path}",
            })

        # Generate model ID
        model_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Load dataset
        X_train, X_test, y_train, y_test, feature_names, target_name = load_dataset(
            dataset_path,
            target_column,
            test_size=test_size,
            random_state=random_state,
        )
        
        # Initialize model
        ml_model = MLModel(
            model_type=model_type,
            task_type=task_type,
            hyperparameters=hyperparameters or {},
        )
        
        ml_model.feature_names = feature_names
        ml_model.target_name = target_name
        
        # Train model
        train_metrics = ml_model.train(
            X_train,
            y_train,
            scale_features=scale_features,
            encode_labels=True,
        )
        
        # Validate on test set
        test_metrics = ml_model.validate(X_test, y_test)
        
        # Save model
        model_path = MODELS_DIR / f"{model_id}.joblib"
        ml_model.save(str(model_path))
        
        # Store in active models
        _register_active_model(model_id, ml_model)
        
        # Get feature importance if available
        feature_importance = ml_model.get_feature_importance()
        
        # Save metadata
        metadata = {
            "model_id": model_id,
            "model_name": model_name,
            "model_type": model_type,
            "task_type": task_type,
            "dataset_path": dataset_path,
            "target_column": target_column,
            "feature_names": feature_names,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "feature_importance": feature_importance,
            "hyperparameters": hyperparameters or {},
            "model_path": str(model_path),
            "trained_at": datetime.now().isoformat(),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
        
        save_model_metadata(model_id, metadata)
        
        result = {
            "success": True,
            "model_id": model_id,
            "model_name": model_name,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "feature_importance": feature_importance,
            "message": f"Model trained successfully: {model_id}",
        }
        
        logger.info(f"Model training completed: {model_id}")
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


@mcp.tool()
async def validate_model(
    model_id: str,
    validation_dataset_path: str,
    target_column: str,
) -> str:
    """
    Validate a trained model on a validation dataset.
    
    Args:
        model_id: ID of the trained model
        validation_dataset_path: Path to validation dataset
        target_column: Name of target column
    
    Returns:
        JSON string with validation results
    """
    try:
        logger.info(f"Validating model: {model_id}")
        
        # Load model
        if model_id in active_models:
            ml_model = active_models[model_id]
        else:
            metadata = load_model_metadata(model_id)
            if not metadata:
                return json.dumps({
                    "success": False,
                    "error": f"Model {model_id} not found",
                })
            
            model_path = metadata["model_path"]
            ml_model = MLModel.load(model_path)
            _register_active_model(model_id, ml_model)
        
        # Load validation data
        import pandas as pd
        df = pd.read_csv(validation_dataset_path)
        
        if target_column not in df.columns:
            return json.dumps({
                "success": False,
                "error": f"Target column '{target_column}' not found",
            })
        
        X_val = df.drop(columns=[target_column]).values
        y_val = df[target_column].values
        
        # Validate
        val_metrics = ml_model.validate(X_val, y_val)
        
        result = {
            "success": True,
            "model_id": model_id,
            "metrics": val_metrics,
            "validation_samples": len(X_val),
        }
        
        logger.info(f"Validation completed: {model_id}")
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error validating model: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


@mcp.tool()
async def get_model_metrics(model_id: str) -> str:
    """
    Get metrics for a trained model.
    
    Args:
        model_id: ID of the model
    
    Returns:
        JSON string with model metrics
    """
    try:
        metadata = load_model_metadata(model_id)
        
        if not metadata:
            return json.dumps({
                "success": False,
                "error": f"Model {model_id} not found",
            })
        
        result = {
            "success": True,
            "model_id": model_id,
            "model_name": metadata.get("model_name"),
            "model_type": metadata.get("model_type"),
            "task_type": metadata.get("task_type"),
            "train_metrics": metadata.get("train_metrics"),
            "test_metrics": metadata.get("test_metrics"),
            "feature_importance": metadata.get("feature_importance"),
            "trained_at": metadata.get("trained_at"),
        }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


@mcp.tool()
async def list_trained_models() -> str:
    """
    List all trained models.
    
    Returns:
        JSON string with list of models
    """
    try:
        models = []
        
        for model_id, metadata in models_registry.items():
            models.append({
                "model_id": model_id,
                "model_name": metadata.get("model_name"),
                "model_type": metadata.get("model_type"),
                "task_type": metadata.get("task_type"),
                "trained_at": metadata.get("trained_at"),
                "test_metrics": metadata.get("test_metrics"),
            })
        
        result = {
            "success": True,
            "total_models": len(models),
            "models": models,
        }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


@mcp.tool()
async def save_model(model_id: str, output_path: Optional[str] = None) -> str:
    """
    Save a trained model to a specific location.
    
    Args:
        model_id: ID of the model to save
        output_path: Optional custom output path
    
    Returns:
        JSON string with save status
    """
    try:
        metadata = load_model_metadata(model_id)
        
        if not metadata:
            return json.dumps({
                "success": False,
                "error": f"Model {model_id} not found",
            })
        
        source_path = metadata["model_path"]
        
        if output_path:
            import shutil
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_path, output_path)
            final_path = str(output_path)
        else:
            final_path = source_path
        
        result = {
            "success": True,
            "model_id": model_id,
            "model_path": final_path,
            "message": f"Model saved to {final_path}",
        }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


@mcp.tool()
async def load_model(model_path: str) -> str:
    """
    Load a model from disk.
    
    Args:
        model_path: Path to the model file
    
    Returns:
        JSON string with load status
    """
    try:
        ml_model = MLModel.load(model_path)
        
        # Generate a new model ID
        model_id = f"loaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store in active models
        _register_active_model(model_id, ml_model)
        
        # Create basic metadata
        metadata = {
            "model_id": model_id,
            "model_type": ml_model.model_type,
            "task_type": ml_model.task_type,
            "model_path": model_path,
            "loaded_at": datetime.now().isoformat(),
        }
        
        save_model_metadata(model_id, metadata)
        
        result = {
            "success": True,
            "model_id": model_id,
            "model_type": ml_model.model_type,
            "task_type": ml_model.task_type,
            "message": f"Model loaded successfully: {model_id}",
        }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


@mcp.tool()
async def predict(model_id: str, features: list[list[float]]) -> str:
    """
    Make predictions using a trained model.
    
    Args:
        model_id: ID of the trained model
        features: List of feature vectors to predict
    
    Returns:
        JSON string with predictions
    """
    try:
        # Load model
        if model_id in active_models:
            ml_model = active_models[model_id]
        else:
            metadata = load_model_metadata(model_id)
            if not metadata:
                return json.dumps({
                    "success": False,
                    "error": f"Model {model_id} not found",
                })
            
            model_path = metadata["model_path"]
            ml_model = MLModel.load(model_path)
            _register_active_model(model_id, ml_model)
        
        # Make predictions
        import numpy as np
        X = np.array(features)
        predictions = ml_model.predict(X)
        
        result = {
            "success": True,
            "model_id": model_id,
            "predictions": predictions.tolist(),
            "num_predictions": len(predictions),
        }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


def main():
    """Run the MCP server."""
    logger.info("Starting ML Training Server (scikit-learn)...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
