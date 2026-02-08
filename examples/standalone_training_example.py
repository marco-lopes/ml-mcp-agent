#!/usr/bin/env python3
"""
Standalone Training Example - Works without MCP

This example demonstrates training a machine learning model
directly using the scikit-learn integration, without relying on MCP servers.
This ensures metrics are always displayed correctly.
"""

import os
import sys
from pathlib import Path

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "servers" / "training_server"))

from models import MLModel, load_dataset
import kagglehub
from sklearn.preprocessing import StandardScaler
import json

def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def print_metrics(metrics: dict, title: str):
    """Print metrics in a formatted way."""
    print(f"📊 {title}:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f} ({value*100:.2f}%)")
        else:
            print(f"   {key}: {value}")

def main():
    """Main training pipeline."""
    
    # Configure Kaggle token
    os.environ["KAGGLE_KEY"] = "KAGGLE_KEY_0f4a6d345afafa670027ce2c7be91807"
    
    print_section("MACHINE LEARNING TRAINING PIPELINE")
    print("This example demonstrates:")
    print("  1. Downloading datasets from Kaggle")
    print("  2. Training ML models with scikit-learn")
    print("  3. Displaying performance metrics")
    print("  4. Analyzing feature importance")
    
    # Step 1: Download dataset
    print_section("STEP 1: Download Dataset from Kaggle")
    
    print("📥 Downloading Iris dataset...")
    dataset_path = kagglehub.dataset_download("uciml/iris")
    csv_file = str(Path(dataset_path) / "Iris.csv")
    print(f"✅ Dataset downloaded: {csv_file}")
    
    # Step 2: Load and prepare data
    print_section("STEP 2: Load and Prepare Data")
    
    print("📂 Loading dataset...")
    X_train, X_test, y_train, y_test, feature_names, target_name = load_dataset(
        csv_file,
        target_column="Species",
        test_size=0.2,
        random_state=42
    )
    
    print(f"✅ Dataset loaded successfully!")
    print(f"\n📊 Dataset Statistics:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Number of features: {len(feature_names)}")
    print(f"   Features: {', '.join(feature_names)}")
    print(f"   Target variable: {target_name}")
    
    # Step 3: Scale features
    print_section("STEP 3: Feature Scaling")
    
    print("⚙️  Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Features scaled")
    
    # Step 4: Train model
    print_section("STEP 4: Train Model")
    
    print("🤖 Training Random Forest Classifier...")
    print("\nHyperparameters:")
    hyperparameters = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "random_state": 42
    }
    for key, value in hyperparameters.items():
        print(f"   {key}: {value}")
    
    model = MLModel(
        model_type="random_forest",
        task_type="classification",
        hyperparameters=hyperparameters
    )
    
    model.train(X_train_scaled, y_train, feature_names)
    print("\n✅ Model trained successfully!")
    
    # Step 5: Evaluate on training data
    print_section("STEP 5: Training Set Performance")
    
    train_metrics = model.validate(X_train_scaled, y_train)
    print_metrics(train_metrics, "Training Metrics")
    
    # Step 6: Evaluate on test data
    print_section("STEP 6: Test Set Performance")
    
    test_metrics = model.validate(X_test_scaled, y_test)
    print_metrics(test_metrics, "Test Metrics")
    
    # Step 7: Feature importance
    print_section("STEP 7: Feature Importance Analysis")
    
    feature_importance = model.get_feature_importance()
    if feature_importance:
        print("🔍 Feature Importance (sorted by importance):\n")
        for i, (feature, importance) in enumerate(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True),
            1
        ):
            bar_length = int(importance * 50)
            bar = "█" * bar_length
            print(f"   {i}. {feature:20s} {importance:.4f} {bar}")
    else:
        print("⚠️  Feature importance not available for this model type")
    
    # Step 8: Save model
    print_section("STEP 8: Save Model")
    
    model_dir = Path("/tmp/ml_models")
    model_dir.mkdir(exist_ok=True)
    model_path = str(model_dir / "iris_classifier.joblib")
    
    saved_path = model.save(model_path)
    print(f"💾 Model saved to: {saved_path}")
    
    # Summary
    print_section("SUMMARY")
    
    print("✅ Training pipeline completed successfully!\n")
    print("📊 Final Results:")
    print(f"   Training Accuracy: {train_metrics.get('val_accuracy', 0):.4f} ({train_metrics.get('val_accuracy', 0)*100:.2f}%)")
    print(f"   Test Accuracy: {test_metrics.get('val_accuracy', 0):.4f} ({test_metrics.get('val_accuracy', 0)*100:.2f}%)")
    print(f"   Test Precision: {test_metrics.get('val_precision', 0):.4f}")
    print(f"   Test Recall: {test_metrics.get('val_recall', 0):.4f}")
    print(f"   Test F1-Score: {test_metrics.get('val_f1_score', 0):.4f}")
    
    if feature_importance:
        top_feature = max(feature_importance.items(), key=lambda x: x[1])
        print(f"\n🏆 Most important feature: {top_feature[0]} ({top_feature[1]:.4f})")
    
    print(f"\n💾 Model saved and ready for deployment!")
    
    print("\n" + "=" * 70)
    print("  🎉 SUCCESS!")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
