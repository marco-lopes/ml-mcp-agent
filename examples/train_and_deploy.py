#!/usr/bin/env python3
"""
Example: Train and Deploy ML Model

This example demonstrates how to use the ML Orchestration Agent
to train and deploy a machine learning model end-to-end.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))

from main import MLAgent


def example_train_and_deploy():
    """Example of training and deploying a model."""
    
    agent = MLAgent()
    
    try:
        # Start the agent
        print("Starting ML Agent...")
        if not agent.start():
            print("Failed to start agent")
            return
        
        print("\n" + "=" * 70)
        print("EXAMPLE 1: Search and Download Dataset from Kaggle")
        print("=" * 70 + "\n")
        
        # Search for a dataset
        search_result = agent.search_datasets(query="iris dataset", max_results=1)
        print("Search Result:")
        print(json.dumps(search_result, indent=2))
        
        if not search_result.get("success") or not search_result.get("datasets"):
            print("\nCould not find dataset. Exiting.")
            return
        
        dataset_ref = search_result["datasets"][0]["ref"]
        print(f"\nFound dataset: {dataset_ref}")
        
        # Download the dataset
        download_result = agent.download_dataset(dataset_ref=dataset_ref)
        print("\nDownload Result:")
        print(json.dumps(download_result, indent=2))
        
        if not download_result.get("success"):
            print("\nFailed to download dataset. Exiting.")
            return
        
        # Find the CSV file path
        dataset_path = None
        for file_info in download_result.get("files", []):
            if file_info["name"].endswith(".csv"):
                dataset_path = file_info["path"]
                break
        
        if not dataset_path:
            print("\nNo CSV file found in the downloaded dataset. Exiting.")
            return
        
        print(f"\nUsing dataset file: {dataset_path}")
        
        print("\n" + "=" * 70)
        print("EXAMPLE 2: Train and Deploy a Model")
        print("=" * 70 + "\n")
        
        # Execute training and deployment pipeline
        pipeline_result = agent.execute_pipeline(
            model_name="iris_classifier_v1",
            model_type="random_forest",
            dataset_path=dataset_path,
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "random_state": 42,
            },
            validation_dataset_path=dataset_path, # Using same for simplicity
            environment="staging",
            min_accuracy=0.80,
        )
        
        print("Pipeline Result:")
        print(json.dumps(pipeline_result, indent=2))
        
        if pipeline_result.get("status") == "completed":
            deployment_id = pipeline_result.get("deployment_id")
            model_id = pipeline_result.get("model_id")
            
            print("\n" + "=" * 70)
            print("EXAMPLE 2: Check Deployment Status")
            print("=" * 70 + "\n")
            
            status = agent.get_deployment_status(deployment_id)
            print("Deployment Status:")
            print(json.dumps(status, indent=2))
            
            print("\n" + "=" * 70)
            print("EXAMPLE 3: List All Trained Models")
            print("=" * 70 + "\n")
            
            models = agent.get_models()
            print("Trained Models:")
            print(json.dumps(models, indent=2))
            
            print("\n" + "=" * 70)
            print("EXAMPLE 4: List All Deployments")
            print("=" * 70 + "\n")
            
            deployments = agent.get_deployments()
            print("Deployments:")
            print(json.dumps(deployments, indent=2))
            
            print("\n" + "=" * 70)
            print("EXAMPLE 5: Update Deployed Model")
            print("=" * 70 + "\n")
            
            # Simulate training a new version
            new_pipeline = agent.execute_pipeline(
                model_name="iris_classifier_v2",
                model_type="random_forest",
                dataset_path="/data/iris_train.csv",
                hyperparameters={
                    "n_estimators": 150,
                    "max_depth": 12,
                    "min_samples_split": 2,
                    "random_state": 42,
                },
                validation_dataset_path="/data/iris_test.csv",
                environment="staging",
                min_accuracy=0.80,
            )
            
            if new_pipeline.get("status") == "completed":
                new_model_id = new_pipeline.get("model_id")
                new_model_path = new_pipeline.get("steps", {}).get("save", {}).get("model_path")
                
                update_result = agent.update_model(
                    deployment_id=deployment_id,
                    new_model_id=new_model_id,
                    new_model_path=new_model_path,
                )
                print("Model Update Result:")
                print(json.dumps(update_result, indent=2))
                
                print("\n" + "=" * 70)
                print("EXAMPLE 6: Rollback Model")
                print("=" * 70 + "\n")
                
                rollback_result = agent.rollback_model(
                    deployment_id=deployment_id,
                    previous_model_id=model_id,
                )
                print("Rollback Result:")
                print(json.dumps(rollback_result, indent=2))
        
        print("\n" + "=" * 70)
        print("EXAMPLE 7: Pipeline Execution History")
        print("=" * 70 + "\n")
        
        history = agent.get_history()
        print("Execution History:")
        print(json.dumps(history, indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        agent.stop()
        print("\nAgent stopped")


if __name__ == "__main__":
    example_train_and_deploy()
