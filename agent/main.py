#!/usr/bin/env python3
"""
ML Orchestration Agent

Main entry point for the ML orchestration agent that coordinates
training and deployment of machine learning models via MCP servers.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

from orchestrator import MLPipelineOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MLAgent:
    """Main ML Orchestration Agent."""
    
    def __init__(self):
        """Initialize the agent."""
        project_root = Path(__file__).parent.parent
        training_server = project_root / "servers" / "training_server" / "main.py"
        deployment_server = project_root / "servers" / "deployment_server" / "main.py"
        kaggle_server = project_root / "servers" / "kaggle_server" / "main.py"
        
        self.orchestrator = MLPipelineOrchestrator(
            str(training_server),
            str(deployment_server),
            str(kaggle_server),
        )
        self.running = False
    
    def start(self) -> bool:
        """Start the agent."""
        logger.info("Starting ML Orchestration Agent...")
        if self.orchestrator.start():
            self.running = True
            logger.info("Agent started successfully")
            return True
        logger.error("Failed to start agent")
        return False
    
    def stop(self) -> None:
        """Stop the agent."""
        logger.info("Stopping ML Orchestration Agent...")
        self.orchestrator.stop()
        self.running = False
        logger.info("Agent stopped")
    
    def execute_pipeline(
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
        Execute a complete ML pipeline.
        
        Args:
            model_name: Name for the model
            model_type: Type of model
            dataset_path: Path to training dataset
            hyperparameters: Model hyperparameters
            validation_dataset_path: Path to validation dataset
            environment: Deployment environment
            min_accuracy: Minimum accuracy threshold
        
        Returns:
            Pipeline execution results
        """
        if not self.running:
            return {"error": "Agent is not running"}
        
        logger.info(f"Executing pipeline for model: {model_name}")
        return self.orchestrator.train_and_deploy(
            model_name=model_name,
            model_type=model_type,
            dataset_path=dataset_path,
            hyperparameters=hyperparameters,
            validation_dataset_path=validation_dataset_path,
            environment=environment,
            min_accuracy=min_accuracy,
        )
    
    def get_models(self) -> dict[str, Any]:
        """Get list of trained models."""
        if not self.running:
            return {"error": "Agent is not running"}
        return self.orchestrator.list_models()
    
    def get_deployments(self) -> dict[str, Any]:
        """Get list of deployments."""
        if not self.running:
            return {"error": "Agent is not running"}
        return self.orchestrator.list_deployments()
    
    def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """Get status of a deployment."""
        if not self.running:
            return {"error": "Agent is not running"}
        return self.orchestrator.get_deployment_status(deployment_id)
    
    def update_model(
        self,
        deployment_id: str,
        new_model_id: str,
        new_model_path: str,
    ) -> dict[str, Any]:
        """Update a deployed model."""
        if not self.running:
            return {"error": "Agent is not running"}
        return self.orchestrator.update_deployed_model(
            deployment_id=deployment_id,
            new_model_id=new_model_id,
            new_model_path=new_model_path,
        )
    
    def rollback_model(
        self,
        deployment_id: str,
        previous_model_id: str,
    ) -> dict[str, Any]:
        """Rollback a deployed model."""
        if not self.running:
            return {"error": "Agent is not running"}
        return self.orchestrator.rollback_deployment(
            deployment_id=deployment_id,
            previous_model_id=previous_model_id,
        )
    
    def get_history(self) -> list[dict[str, Any]]:
        """Get pipeline execution history."""
        return self.orchestrator.get_pipeline_history()
    
    def search_datasets(self, query: str, max_results: int = 10) -> dict[str, Any]:
        """Search for datasets on Kaggle."""
        if not self.running:
            return {"error": "Agent is not running"}
        return self.orchestrator.search_kaggle_datasets(query, max_results)
    
    def download_dataset(self, dataset_ref: str, output_dir: str = None) -> dict[str, Any]:
        """Download a dataset from Kaggle."""
        if not self.running:
            return {"error": "Agent is not running"}
        return self.orchestrator.download_kaggle_dataset(dataset_ref, output_dir)
    
    def list_dataset_files(self, dataset_ref: str) -> dict[str, Any]:
        """List files in a Kaggle dataset."""
        if not self.running:
            return {"error": "Agent is not running"}
        return self.orchestrator.list_kaggle_dataset_files(dataset_ref)
    
    def list_downloaded_datasets(self) -> dict[str, Any]:
        """List downloaded Kaggle datasets."""
        if not self.running:
            return {"error": "Agent is not running"}
        return self.orchestrator.list_downloaded_kaggle_datasets()
    
    def train_model(
        self,
        model_name: str,
        model_type: str,
        dataset_path: str,
        target_column: str,
        task_type: str = "classification",
        hyperparameters: dict[str, Any] = None,
        test_size: float = 0.2,
        scale_features: bool = True,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Train a machine learning model.
        
        Args:
            model_name: Name for the model
            model_type: Type of model (random_forest, logistic_regression, etc.)
            dataset_path: Path to the dataset CSV file
            target_column: Name of the target column
            task_type: 'classification' or 'regression'
            hyperparameters: Model hyperparameters
            test_size: Proportion of data for testing
            scale_features: Whether to scale features
            random_state: Random seed
        
        Returns:
            Training results with metrics
        """
        if not self.running:
            return {"error": "Agent is not running"}
        
        return self.orchestrator.training_client.call_tool(
            "train_model",
            model_name=model_name,
            model_type=model_type,
            dataset_path=dataset_path,
            target_column=target_column,
            task_type=task_type,
            hyperparameters=hyperparameters or {},
            test_size=test_size,
            scale_features=scale_features,
            random_state=random_state,
        )
    
    def validate_model(
        self,
        model_id: str,
        validation_dataset_path: str,
        target_column: str,
    ) -> dict[str, Any]:
        """Validate a trained model.
        
        Args:
            model_id: ID of the trained model
            validation_dataset_path: Path to validation dataset
            target_column: Name of target column
        
        Returns:
            Validation results with metrics
        """
        if not self.running:
            return {"error": "Agent is not running"}
        
        return self.orchestrator.training_client.call_tool(
            "validate_model",
            model_id=model_id,
            validation_dataset_path=validation_dataset_path,
            target_column=target_column,
        )
    
    def get_model_metrics(self, model_id: str) -> dict[str, Any]:
        """Get metrics for a trained model.
        
        Args:
            model_id: ID of the model
        
        Returns:
            Model metrics
        """
        if not self.running:
            return {"error": "Agent is not running"}
        
        return self.orchestrator.training_client.call_tool(
            "get_model_metrics",
            model_id=model_id,
        )
    
    def list_models(self) -> dict[str, Any]:
        """List all trained models.
        
        Returns:
            List of trained models
        """
        if not self.running:
            return {"error": "Agent is not running"}
        
        return self.orchestrator.training_client.call_tool(
            "list_trained_models",
        )
    
    def deploy_model(
        self,
        model_id: str,
        environment: str = "staging",
        endpoint_name: str = None,
    ) -> dict[str, Any]:
        """Deploy a trained model.
        
        Args:
            model_id: ID of the model to deploy
            environment: Deployment environment (staging, production)
            endpoint_name: Name for the endpoint
        
        Returns:
            Deployment results
        """
        if not self.running:
            return {"error": "Agent is not running"}
        
        return self.orchestrator.deployment_client.call_tool(
            "deploy_model",
            model_id=model_id,
            environment=environment,
            endpoint_name=endpoint_name or f"{model_id}-endpoint",
        )
    
    def list_deployments(self) -> dict[str, Any]:
        """List all deployments.
        
        Returns:
            List of deployments
        """
        if not self.running:
            return {"error": "Agent is not running"}
        
        return self.orchestrator.deployment_client.call_tool(
            "list_deployments",
        )


def main():
    """Main entry point."""
    agent = MLAgent()
    
    try:
        if not agent.start():
            sys.exit(1)
        
        # Example: Execute a training and deployment pipeline
        logger.info("=" * 60)
        logger.info("Executing example ML pipeline")
        logger.info("=" * 60)
        
        result = agent.execute_pipeline(
            model_name="iris_classifier",
            model_type="random_forest",
            dataset_path="/data/iris_train.csv",
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
            },
            validation_dataset_path="/data/iris_test.csv",
            environment="staging",
            min_accuracy=0.85,
        )
        
        logger.info("Pipeline Result:")
        logger.info(json.dumps(result, indent=2))
        
        # Get models
        logger.info("\n" + "=" * 60)
        logger.info("Listing trained models")
        logger.info("=" * 60)
        models = agent.get_models()
        logger.info(json.dumps(models, indent=2))
        
        # Get deployments
        logger.info("\n" + "=" * 60)
        logger.info("Listing deployments")
        logger.info("=" * 60)
        deployments = agent.get_deployments()
        logger.info(json.dumps(deployments, indent=2))
        
        # Get history
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline execution history")
        logger.info("=" * 60)
        history = agent.get_history()
        logger.info(json.dumps(history, indent=2))
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
    finally:
        agent.stop()


if __name__ == "__main__":
    main()
