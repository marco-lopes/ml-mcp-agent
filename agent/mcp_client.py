#!/usr/bin/env python3
"""
MCP Client for communicating with ML Training and Deployment servers

This module provides a client interface to interact with MCP servers
for training and deploying machine learning models.
"""

import json
import logging
import subprocess
from typing import Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for communicating with MCP servers via stdio transport."""
    
    def __init__(self, server_script: str, server_name: str):
        """
        Initialize MCP client.
        
        Args:
            server_script: Path to the server script to run
            server_name: Name of the server for logging
        """
        self.server_script = server_script
        self.server_name = server_name
        self.process: Optional[subprocess.Popen] = None
    
    def start(self) -> bool:
        """Start the MCP server process."""
        try:
            logger.info(f"Starting {self.server_name}...")
            process = subprocess.Popen(
                ["python3", self.server_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.process = process
            logger.info(f"{self.server_name} started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start {self.server_name}: {str(e)}")
            return False
    
    def stop(self) -> None:
        """Stop the MCP server process."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                logger.info(f"{self.server_name} stopped")
            except subprocess.TimeoutExpired:
                self.process.kill()
                logger.warning(f"{self.server_name} force killed")
            except Exception as e:
                logger.error(f"Error stopping {self.server_name}: {str(e)}")
    
    def call_tool(self, tool_name: str, **kwargs) -> dict[str, Any]:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            **kwargs: Arguments to pass to the tool
        
        Returns:
            Dictionary with tool result
        """
        if not self.process:
            return {"error": f"{self.server_name} is not running"}
        
        try:
            # Create JSON-RPC request
            request = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": kwargs,
                },
                "id": 1,
            }

            # Send request to server
            try:
                request_json = json.dumps(request) + "\n"
                self.process.stdin.write(request_json)
                self.process.stdin.flush()
            except (BrokenPipeError, IOError) as e:
                logger.error(f"Communication error with {self.server_name}: {e}")
                return {"error": f"Communication error with {self.server_name}: server may have crashed"}

            # Read response from server
            response_line = self.process.stdout.readline()
            if not response_line:
                return {"error": f"No response from {self.server_name}: server may have terminated"}

            try:
                response = json.loads(response_line)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from {self.server_name}: {response_line}")
                return {"error": f"Invalid JSON response from {self.server_name}"}

            if "result" in response:
                # Parse the result if it's a JSON string
                result = response["result"]
                if isinstance(result, str):
                    try:
                        return json.loads(result)
                    except json.JSONDecodeError:
                        return {"result": result}
                return result
            elif "error" in response:
                return {"error": response["error"]}

            return {"error": "No response from server"}

        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {str(e)}")
            return {"error": str(e)}


class TrainingClient(MCPClient):
    """Client for ML Training Server."""
    
    def __init__(self, server_script: str):
        """Initialize training client."""
        super().__init__(server_script, "ML Training Server")
    
    def train_model(
        self,
        model_name: str,
        model_type: str,
        dataset_path: str,
        hyperparameters: dict[str, Any],
        test_size: float = 0.2,
    ) -> dict[str, Any]:
        """Train a model."""
        return self.call_tool(
            "train_model",
            model_name=model_name,
            model_type=model_type,
            dataset_path=dataset_path,
            hyperparameters=hyperparameters,
            test_size=test_size,
        )
    
    def validate_model(
        self,
        model_id: str,
        validation_dataset_path: str,
        target_column: str = "",
    ) -> dict[str, Any]:
        """Validate a trained model."""
        return self.call_tool(
            "validate_model",
            model_id=model_id,
            validation_dataset_path=validation_dataset_path,
            target_column=target_column,
        )
    
    def get_model_metrics(self, model_id: str) -> dict[str, Any]:
        """Get metrics for a model."""
        return self.call_tool("get_model_metrics", model_id=model_id)
    
    def list_trained_models(self) -> dict[str, Any]:
        """List all trained models."""
        return self.call_tool("list_trained_models")
    
    def save_model(self, model_id: str, output_path: str) -> dict[str, Any]:
        """Save a model."""
        return self.call_tool(
            "save_model",
            model_id=model_id,
            output_path=output_path,
        )
    
    def load_model(self, model_path: str) -> dict[str, Any]:
        """Load a model."""
        return self.call_tool("load_model", model_path=model_path)


class DeploymentClient(MCPClient):
    """Client for ML Deployment Server."""
    
    def __init__(self, server_script: str):
        """Initialize deployment client."""
        super().__init__(server_script, "ML Deployment Server")
    
    def deploy_model(
        self,
        model_id: str,
        model_path: str,
        environment: str = "staging",
        replicas: int = 1,
    ) -> dict[str, Any]:
        """Deploy a model."""
        return self.call_tool(
            "deploy_model",
            model_id=model_id,
            model_path=model_path,
            environment=environment,
            replicas=replicas,
        )
    
    def create_endpoint(
        self,
        deployment_id: str,
        endpoint_name: str,
        port: int = 8000,
    ) -> dict[str, Any]:
        """Create an inference endpoint."""
        return self.call_tool(
            "create_endpoint",
            deployment_id=deployment_id,
            endpoint_name=endpoint_name,
            port=port,
        )
    
    def test_endpoint(
        self,
        endpoint_name: str,
        test_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Test an endpoint."""
        return self.call_tool(
            "test_endpoint",
            endpoint_name=endpoint_name,
            test_data=test_data,
        )
    
    def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """Get deployment status."""
        return self.call_tool(
            "get_deployment_status",
            deployment_id=deployment_id,
        )
    
    def list_deployments(self) -> dict[str, Any]:
        """List all deployments."""
        return self.call_tool("list_deployments")
    
    def rollback_deployment(
        self,
        deployment_id: str,
        previous_version: str,
    ) -> dict[str, Any]:
        """Rollback a deployment."""
        return self.call_tool(
            "rollback_deployment",
            deployment_id=deployment_id,
            previous_version=previous_version,
        )
    
    def update_model(
        self,
        deployment_id: str,
        new_model_path: str,
        new_model_id: str,
    ) -> dict[str, Any]:
        """Update a deployed model."""
        return self.call_tool(
            "update_model",
            deployment_id=deployment_id,
            new_model_path=new_model_path,
            new_model_id=new_model_id,
        )


class KaggleClient(MCPClient):
    """Client for Kaggle Dataset Server."""
    
    def __init__(self, server_script: str):
        """Initialize Kaggle client."""
        super().__init__(server_script, "Kaggle Dataset Server")
    
    def search_datasets(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "hottest",
    ) -> dict[str, Any]:
        """Search for datasets on Kaggle."""
        return self.call_tool(
            "search_datasets",
            query=query,
            max_results=max_results,
            sort_by=sort_by,
        )
    
    def download_dataset(
        self,
        dataset_ref: str,
        output_dir: Optional[str] = None,
        unzip: bool = True,
    ) -> dict[str, Any]:
        """Download a dataset from Kaggle."""
        return self.call_tool(
            "download_dataset",
            dataset_ref=dataset_ref,
            output_dir=output_dir,
            unzip=unzip,
        )
    
    def list_dataset_files(self, dataset_ref: str) -> dict[str, Any]:
        """List files in a Kaggle dataset."""
        return self.call_tool("list_dataset_files", dataset_ref=dataset_ref)
    
    def get_dataset_metadata(self, dataset_ref: str) -> dict[str, Any]:
        """Get metadata for a Kaggle dataset."""
        return self.call_tool("get_dataset_metadata", dataset_ref=dataset_ref)
    
    def list_downloaded_datasets(self) -> dict[str, Any]:
        """List all downloaded datasets."""
        return self.call_tool("list_downloaded_datasets")
    
    def get_downloaded_dataset_path(self, dataset_ref: str) -> dict[str, Any]:
        """Get local path of a downloaded dataset."""
        return self.call_tool(
            "get_downloaded_dataset_path",
            dataset_ref=dataset_ref,
        )
