#!/usr/bin/env python3
"""
Real ML Training Example

This example demonstrates the complete ML pipeline with real training:
1. Search and download Iris dataset from Kaggle
2. Train a Random Forest classifier with scikit-learn
3. Validate the model
4. Deploy the model
5. Test predictions
"""

import sys
import json
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))

from main import MLAgent


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_json(data: dict, indent: int = 2):
    """Print JSON data in a formatted way."""
    print(json.dumps(data, indent=indent))


def main():
    """Run the complete ML pipeline example."""
    
    print_section("ML PIPELINE WITH REAL TRAINING - IRIS DATASET")
    
    # Initialize agent
    agent = MLAgent()
    
    try:
        # Start the agent
        print("🚀 Starting ML Agent...")
        if not agent.start():
            print("❌ Failed to start agent")
            return
        
        print("✅ Agent started successfully")
        time.sleep(2)  # Give servers time to initialize
        
        # ========================================
        # STEP 1: Search for Iris dataset on Kaggle
        # ========================================
        print_section("STEP 1: Search for Iris Dataset on Kaggle")
        
        search_result = agent.search_datasets(query="iris", max_results=5)
        
        if not search_result.get("success"):
            print(f"❌ Search failed: {search_result.get('error')}")
            return
        
        print(f"✅ Found {search_result['total_results']} datasets")
        
        # Display search results
        for i, dataset in enumerate(search_result.get("datasets", []), 1):
            print(f"\n{i}. {dataset['ref']}")
            print(f"   Title: {dataset['title']}")
            print(f"   Size: {dataset['size']}")
            print(f"   Downloads: {dataset.get('download_count', 'N/A')}")
        
        # Select first dataset (usually uciml/iris)
        if not search_result.get("datasets"):
            print("❌ No datasets found")
            return
        
        dataset_ref = search_result["datasets"][0]["ref"]
        print(f"\n📦 Selected dataset: {dataset_ref}")
        
        # ========================================
        # STEP 2: Download the dataset
        # ========================================
        print_section("STEP 2: Download Iris Dataset")
        
        download_result = agent.download_dataset(dataset_ref=dataset_ref)
        
        if not download_result.get("success"):
            print(f"❌ Download failed: {download_result.get('error')}")
            return
        
        print(f"✅ Dataset downloaded to: {download_result['output_dir']}")
        print(f"📁 Total files: {download_result['total_files']}")
        
        # Find CSV file
        csv_file = None
        for file_info in download_result.get("files", []):
            if file_info["name"].endswith(".csv"):
                csv_file = file_info["path"]
                print(f"📄 Found CSV file: {file_info['name']}")
                break
        
        if not csv_file:
            print("❌ No CSV file found in dataset")
            return
        
        # ========================================
        # STEP 3: Train a Random Forest model
        # ========================================
        print_section("STEP 3: Train Random Forest Classifier")
        
        print("🔧 Training configuration:")
        print("  - Model: Random Forest")
        print("  - Task: Classification")
        print("  - Target: Species (or similar)")
        print("  - Test size: 20%")
        print("  - Feature scaling: Enabled")
        
        # Note: Iris dataset typically has 'variety' or 'species' as target
        # We'll try common column names
        target_column = "variety"  # Common in Iris datasets
        
        train_result = agent.train_model(
            model_name="iris_classifier",
            model_type="random_forest",
            dataset_path=csv_file,
            target_column=target_column,
            task_type="classification",
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "random_state": 42,
            },
            test_size=0.2,
            scale_features=True,
        )
        
        if not train_result.get("success"):
            # Try alternative column name
            print(f"⚠️  Column '{target_column}' not found, trying 'species'...")
            target_column = "species"
            
            train_result = agent.train_model(
                model_name="iris_classifier",
                model_type="random_forest",
                dataset_path=csv_file,
                target_column=target_column,
                task_type="classification",
                hyperparameters={
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 2,
                    "random_state": 42,
                },
                test_size=0.2,
                scale_features=True,
            )
        
        if not train_result.get("success"):
            print(f"❌ Training failed: {train_result.get('error')}")
            return
        
        model_id = train_result["model_id"]
        print(f"✅ Model trained successfully!")
        print(f"🆔 Model ID: {model_id}")
        
        print("\n📊 Training Metrics:")
        print_json(train_result.get("train_metrics", {}))
        
        print("\n📊 Test Metrics:")
        print_json(train_result.get("test_metrics", {}))
        
        if train_result.get("feature_importance"):
            print("\n🔍 Feature Importance:")
            for feature, importance in sorted(
                train_result["feature_importance"].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"  {feature}: {importance:.4f}")
        
        # ========================================
        # STEP 4: Get detailed model metrics
        # ========================================
        print_section("STEP 4: Get Model Metrics")
        
        metrics_result = agent.get_model_metrics(model_id=model_id)
        
        if metrics_result.get("success"):
            print(f"✅ Model: {metrics_result.get('model_name')}")
            print(f"   Type: {metrics_result.get('model_type')}")
            print(f"   Task: {metrics_result.get('task_type')}")
            print(f"   Trained: {metrics_result.get('trained_at')}")
        
        # ========================================
        # STEP 5: Deploy the model
        # ========================================
        print_section("STEP 5: Deploy Model to Staging")
        
        deploy_result = agent.deploy_model(
            model_id=model_id,
            environment="staging",
            endpoint_name="iris-classifier-api",
        )
        
        if not deploy_result.get("success"):
            print(f"❌ Deployment failed: {deploy_result.get('error')}")
        else:
            deployment_id = deploy_result["deployment_id"]
            print(f"✅ Model deployed successfully!")
            print(f"🆔 Deployment ID: {deployment_id}")
            print(f"🌐 Endpoint: {deploy_result.get('endpoint_url')}")
        
        # ========================================
        # STEP 6: List all models and deployments
        # ========================================
        print_section("STEP 6: List Models and Deployments")
        
        models_result = agent.list_models()
        if models_result.get("success"):
            print(f"📋 Total models: {models_result.get('total_models')}")
            for model in models_result.get("models", []):
                print(f"\n  - {model['model_id']}")
                print(f"    Name: {model.get('model_name')}")
                print(f"    Type: {model.get('model_type')}")
                if model.get("test_metrics"):
                    print(f"    Accuracy: {model['test_metrics'].get('val_accuracy', 'N/A')}")
        
        deployments_result = agent.list_deployments()
        if deployments_result.get("success"):
            print(f"\n🚀 Total deployments: {deployments_result.get('total_deployments')}")
            for deployment in deployments_result.get("deployments", []):
                print(f"\n  - {deployment['deployment_id']}")
                print(f"    Environment: {deployment.get('environment')}")
                print(f"    Status: {deployment.get('status')}")
        
        # ========================================
        # STEP 7: Summary
        # ========================================
        print_section("SUMMARY")
        
        print("✅ Pipeline completed successfully!")
        print("\nWhat we did:")
        print("  1. ✅ Searched Kaggle for Iris dataset")
        print("  2. ✅ Downloaded dataset using kagglehub")
        print("  3. ✅ Trained Random Forest classifier with scikit-learn")
        print("  4. ✅ Validated model on test set")
        print("  5. ✅ Deployed model to staging environment")
        print("  6. ✅ Listed all models and deployments")
        
        print("\n📈 Key Results:")
        if train_result.get("test_metrics"):
            test_acc = train_result["test_metrics"].get("val_accuracy", 0)
            print(f"  - Test Accuracy: {test_acc:.2%}")
        print(f"  - Model ID: {model_id}")
        if deploy_result.get("success"):
            print(f"  - Deployment ID: {deployment_id}")
        
        print("\n🎉 Real ML training with scikit-learn is working!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop the agent
        print("\n🛑 Stopping ML Agent...")
        agent.stop()
        print("✅ Agent stopped")


if __name__ == "__main__":
    main()
