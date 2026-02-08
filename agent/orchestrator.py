#!/usr/bin/env python3
"""
ML Pipeline Orchestrator

Orchestrates the complete workflow of training and deploying ML models
using MCP servers for training and deployment operations.
"""

import json
import logging
from typing import Any, Optional
from pathlib import Path
from datetime import datetime

from mcp_client import TrainingClient, DeploymentClient, KaggleClient

logger = logging.getLogger(__name__)


class MLPipelineOrchestrator:
    """Orchestrates ML model training and deployment pipeline."""
    
    def __init__(
        self,
        training_server_path: str,
        deployment_server_path: str,
        kaggle_server_path: str,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            training_server_path: Path to training server script
            deployment_server_path: Path to deployment server script
            kaggle_server_path: Path to Kaggle server script
        """
        self.training_client = TrainingClient(training_server_path)
        self.deployment_client = DeploymentClient(deployment_server_path)
        self.kaggle_client = KaggleClient(kaggle_server_path)
        self.pipeline_history: list[dict[str, Any]] = []
    
    def start(self) -> bool:
        """Start all MCP servers."""
        logger.info("Starting ML Pipeline Orchestrator...")
        
        if not self.training_client.start():
            logger.error("Failed to start training server")
            return False
        
        if not self.deployment_client.start():
            logger.error("Failed to start deployment server")
            self.training_client.stop()
            return False
        
        if not self.kaggle_client.start():
            logger.error("Failed to start Kaggle server")
            self.training_client.stop()
            self.deployment_client.stop()
            return False
        
        logger.info("All servers started successfully")
        return True
    
    def stop(self) -> None:
        """Stop all MCP servers."""
        logger.info("Stopping ML Pipeline Orchestrator...")
        self.training_client.stop()
        self.deployment_client.stop()
        self.kaggle_client.stop()
        logger.info("All servers stopped")
    
    def train_and_deploy(
        self,
        model_name: str,
        model_type: str,
        dataset_path: str,
        hyperparameters: dict[str, Any],
        validation_dataset_path: str,
        environment: str = "staging",
        min_accuracy: float = 0.85,
    ) -> dict[str, Any]:
        """
        Complete pipeline: train model, validate, and deploy.
        
        Args:
            model_name: Name for the model
            model_type: Type of model (e.g., 'random_forest')
            dataset_path: Path to training dataset
            hyperparameters: Model hyperparameters
            validation_dataset_path: Path to validation dataset
            environment: Deployment environment ('staging' or 'production')
            min_accuracy: Minimum accuracy threshold for deployment
        
        Returns:
            Dictionary with pipeline results
        """
        pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting pipeline: {pipeline_id}")
        
        pipeline_result = {
            "pipeline_id": pipeline_id,
            "model_name": model_name,
            "status": "in_progress",
            "steps": {},
        }
        
        try:
            # Step 1: Train model
            logger.info("Step 1: Training model...")
            train_result = self.training_client.train_model(
                model_name=model_name,
                model_type=model_type,
                dataset_path=dataset_path,
                hyperparameters=hyperparameters,
            )
            
            if not train_result.get("success"):
                pipeline_result["status"] = "failed"
                pipeline_result["steps"]["training"] = train_result
                return pipeline_result
            
            model_id = train_result.get("model_id")
            pipeline_result["steps"]["training"] = {
                "status": "completed",
                "model_id": model_id,
                "metrics": train_result.get("metrics"),
            }
            logger.info(f"Model trained: {model_id}")
            
            # Step 2: Validate model
            logger.info("Step 2: Validating model...")
            validation_result = self.training_client.validate_model(
                model_id=model_id,
                validation_dataset_path=validation_dataset_path,
            )
            
            if not validation_result.get("success"):
                pipeline_result["status"] = "failed"
                pipeline_result["steps"]["validation"] = validation_result
                return pipeline_result
            
            accuracy = validation_result.get("metrics", {}).get("accuracy", 0)
            pipeline_result["steps"]["validation"] = {
                "status": "completed",
                "metrics": validation_result.get("metrics"),
            }
            logger.info(f"Model validated with accuracy: {accuracy}")
            
            # Check accuracy threshold
            if accuracy < min_accuracy:
                pipeline_result["status"] = "failed"
                pipeline_result["error"] = (
                    f"Model accuracy {accuracy} below threshold {min_accuracy}"
                )
                return pipeline_result
            
            # Step 3: Save model
            logger.info("Step 3: Saving model...")
            model_path = f"/tmp/ml_models/{model_id}.pkl"
            save_result = self.training_client.save_model(
                model_id=model_id,
                output_path=model_path,
            )
            
            if not save_result.get("success"):
                pipeline_result["status"] = "failed"
                pipeline_result["steps"]["save"] = save_result
                return pipeline_result
            
            pipeline_result["steps"]["save"] = {
                "status": "completed",
                "model_path": model_path,
            }
            logger.info(f"Model saved: {model_path}")
            
            # Step 4: Deploy model
            logger.info(f"Step 4: Deploying model to {environment}...")
            deploy_result = self.deployment_client.deploy_model(
                model_id=model_id,
                model_path=model_path,
                environment=environment,
                replicas=1,
            )
            
            if not deploy_result.get("success"):
                pipeline_result["status"] = "failed"
                pipeline_result["steps"]["deployment"] = deploy_result
                return pipeline_result
            
            deployment_id = deploy_result.get("deployment_id")
            endpoint_url = deploy_result.get("endpoint_url")
            pipeline_result["steps"]["deployment"] = {
                "status": "completed",
                "deployment_id": deployment_id,
                "endpoint_url": endpoint_url,
            }
            logger.info(f"Model deployed: {deployment_id}")
            
            # Step 5: Create endpoint
            logger.info("Step 5: Creating inference endpoint...")
            endpoint_result = self.deployment_client.create_endpoint(
                deployment_id=deployment_id,
                endpoint_name=f"{model_name}_endpoint",
                port=8000,
            )
            
            if not endpoint_result.get("success"):
                pipeline_result["status"] = "failed"
                pipeline_result["steps"]["endpoint"] = endpoint_result
                return pipeline_result
            
            pipeline_result["steps"]["endpoint"] = {
                "status": "completed",
                "endpoint_name": endpoint_result.get("endpoint_name"),
                "endpoint_url": endpoint_result.get("endpoint_url"),
            }
            logger.info(f"Endpoint created: {endpoint_result.get('endpoint_url')}")
            
            # Step 6: Test endpoint
            logger.info("Step 6: Testing endpoint...")
            test_data = {"feature_1": 1.0, "feature_2": 2.0}
            test_result = self.deployment_client.test_endpoint(
                endpoint_name=f"{model_name}_endpoint",
                test_data=test_data,
            )
            
            if not test_result.get("success"):
                pipeline_result["status"] = "failed"
                pipeline_result["steps"]["testing"] = test_result
                return pipeline_result
            
            pipeline_result["steps"]["testing"] = {
                "status": "completed",
                "prediction": test_result.get("prediction"),
                "confidence": test_result.get("confidence"),
                "latency_ms": test_result.get("latency_ms"),
            }
            logger.info("Endpoint test passed")
            
            # Pipeline completed successfully
            pipeline_result["status"] = "completed"
            pipeline_result["model_id"] = model_id
            pipeline_result["deployment_id"] = deployment_id
            pipeline_result["endpoint_url"] = endpoint_url
            
            logger.info(f"Pipeline completed successfully: {pipeline_id}")
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            pipeline_result["status"] = "failed"
            pipeline_result["error"] = str(e)
        
        self.pipeline_history.append(pipeline_result)
        return pipeline_result
    
    def list_models(self) -> dict[str, Any]:
        """List all trained models."""
        logger.info("Listing trained models...")
        return self.training_client.list_trained_models()
    
    def list_deployments(self) -> dict[str, Any]:
        """List all deployments."""
        logger.info("Listing deployments...")
        return self.deployment_client.list_deployments()
    
    def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """Get status of a deployment."""
        logger.info(f"Getting status for deployment: {deployment_id}")
        return self.deployment_client.get_deployment_status(deployment_id)
    
    def update_deployed_model(
        self,
        deployment_id: str,
        new_model_id: str,
        new_model_path: str,
    ) -> dict[str, Any]:
        """Update a deployed model with a new version."""
        logger.info(f"Updating model in deployment: {deployment_id}")
        return self.deployment_client.update_model(
            deployment_id=deployment_id,
            new_model_path=new_model_path,
            new_model_id=new_model_id,
        )
    
    def rollback_deployment(
        self,
        deployment_id: str,
        previous_model_id: str,
    ) -> dict[str, Any]:
        """Rollback a deployment to previous version."""
        logger.info(f"Rolling back deployment: {deployment_id}")
        return self.deployment_client.rollback_deployment(
            deployment_id=deployment_id,
            previous_version=previous_model_id,
        )
    
    def get_pipeline_history(self) -> list[dict[str, Any]]:
        """Get history of executed pipelines."""
        return self.pipeline_history
    
    def search_kaggle_datasets(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "hottest",
    ) -> dict[str, Any]:
        """Search for datasets on Kaggle."""
        logger.info(f"Searching Kaggle datasets: {query}")
        return self.kaggle_client.search_datasets(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
        )
    
    def download_kaggle_dataset(
        self,
        dataset_ref: str,
        output_dir: Optional[str] = None,
    ) -> dict[str, Any]:
        """Download a dataset from Kaggle."""
        logger.info(f"Downloading Kaggle dataset: {dataset_ref}")
        return self.kaggle_client.download_dataset(
            dataset_ref=dataset_ref,
            output_dir=output_dir,
            unzip=True,
        )
    
    def list_kaggle_dataset_files(self, dataset_ref: str) -> dict[str, Any]:
        """List files in a Kaggle dataset."""
        logger.info(f"Listing files for dataset: {dataset_ref}")
        return self.kaggle_client.list_dataset_files(dataset_ref)
    
    def get_kaggle_dataset_metadata(self, dataset_ref: str) -> dict[str, Any]:
        """Get metadata for a Kaggle dataset."""
        logger.info(f"Getting metadata for dataset: {dataset_ref}")
        return self.kaggle_client.get_dataset_metadata(dataset_ref)
    
    def list_downloaded_kaggle_datasets(self) -> dict[str, Any]:
        """List all downloaded Kaggle datasets."""
        logger.info("Listing downloaded Kaggle datasets")
        return self.kaggle_client.list_downloaded_datasets()
    
    def get_kaggle_dataset_path(self, dataset_ref: str) -> dict[str, Any]:
        """Get local path of a downloaded Kaggle dataset."""
        logger.info(f"Getting path for dataset: {dataset_ref}")
        return self.kaggle_client.get_downloaded_dataset_path(dataset_ref)
