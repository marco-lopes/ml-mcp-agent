#!/usr/bin/env python3
"""
MCP Server for Machine Learning Model Deployment

This server provides tools for deploying trained models, managing endpoints,
and handling inference requests. It communicates via the Model Context Protocol.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("ml-deployment-server")

# Storage for deployment metadata
DEPLOYMENTS_DIR = Path("/tmp/ml_deployments")
DEPLOYMENTS_DIR.mkdir(exist_ok=True)

# In-memory storage for deployment metadata
deployments_registry: dict[str, dict[str, Any]] = {}


def save_deployment_metadata(deployment_id: str, metadata: dict[str, Any]) -> None:
    """Save deployment metadata to registry."""
    deployments_registry[deployment_id] = metadata
    metadata_file = DEPLOYMENTS_DIR / f"{deployment_id}_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Deployment metadata saved: {deployment_id}")


def load_deployment_metadata(deployment_id: str) -> Optional[dict[str, Any]]:
    """Load deployment metadata from registry."""
    if deployment_id in deployments_registry:
        return deployments_registry[deployment_id]
    
    metadata_file = DEPLOYMENTS_DIR / f"{deployment_id}_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            return json.load(f)
    return None


@mcp.tool()
async def deploy_model(
    model_id: str,
    model_path: str,
    environment: str = "staging",
    replicas: int = 1,
) -> str:
    """
    Deploy a trained model to a specified environment.
    
    Args:
        model_id: ID of the model to deploy
        model_path: Path to the saved model file
        environment: Deployment environment ('staging' or 'production')
        replicas: Number of model replicas to deploy
    
    Returns:
        JSON string with deployment details
    """
    try:
        logger.info(f"Starting deployment for model: {model_id}")
        
        # Generate unique deployment ID
        deployment_id = f"deploy_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create deployment metadata
        deployment_metadata = {
            "deployment_id": deployment_id,
            "model_id": model_id,
            "model_path": model_path,
            "environment": environment,
            "replicas": replicas,
            "status": "deployed",
            "created_at": datetime.now().isoformat(),
            "endpoint_url": f"http://localhost:8000/predict/{deployment_id}",
            "health_status": "healthy",
        }
        
        save_deployment_metadata(deployment_id, deployment_metadata)
        
        result = {
            "success": True,
            "deployment_id": deployment_id,
            "model_id": model_id,
            "message": f"Model deployed successfully to {environment}",
            "endpoint_url": deployment_metadata["endpoint_url"],
            "environment": environment,
            "replicas": replicas,
            "status": "deployed",
        }
        
        logger.info(f"Deployment completed: {deployment_id}")
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error during model deployment: {str(e)}")
        error_result = {
            "success": False,
            "error": str(e),
            "message": "Failed to deploy model",
        }
        return json.dumps(error_result)


@mcp.tool()
async def create_endpoint(
    deployment_id: str,
    endpoint_name: str,
    port: int = 8000,
) -> str:
    """
    Create an inference endpoint for a deployed model.
    
    Args:
        deployment_id: ID of the deployment
        endpoint_name: Name for the endpoint
        port: Port number for the endpoint
    
    Returns:
        JSON string with endpoint details
    """
    try:
        logger.info(f"Creating endpoint for deployment: {deployment_id}")
        
        metadata = load_deployment_metadata(deployment_id)
        if not metadata:
            return json.dumps({
                "success": False,
                "error": f"Deployment {deployment_id} not found",
            })
        
        endpoint_url = f"http://localhost:{port}/predict/{endpoint_name}"
        
        endpoint_metadata = {
            "endpoint_name": endpoint_name,
            "deployment_id": deployment_id,
            "endpoint_url": endpoint_url,
            "port": port,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "request_count": 0,
            "error_count": 0,
            "average_latency_ms": 0.0,
        }
        
        # Update deployment metadata with endpoint info
        metadata["endpoint"] = endpoint_metadata
        save_deployment_metadata(deployment_id, metadata)
        
        result = {
            "success": True,
            "endpoint_name": endpoint_name,
            "endpoint_url": endpoint_url,
            "deployment_id": deployment_id,
            "status": "active",
            "message": f"Endpoint '{endpoint_name}' created successfully",
        }
        
        logger.info(f"Endpoint created: {endpoint_name}")
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error creating endpoint: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


@mcp.tool()
async def test_endpoint(
    endpoint_name: str,
    test_data: dict[str, Any],
) -> str:
    """
    Test an inference endpoint with sample data.
    
    Args:
        endpoint_name: Name of the endpoint to test
        test_data: Test input data for the model
    
    Returns:
        JSON string with test results and predictions
    """
    try:
        logger.info(f"Testing endpoint: {endpoint_name}")
        
        # Simulate endpoint test
        test_result = {
            "success": True,
            "endpoint_name": endpoint_name,
            "test_timestamp": datetime.now().isoformat(),
            "input": test_data,
            "prediction": 0.87,  # Simulated prediction
            "confidence": 0.92,  # Simulated confidence
            "latency_ms": 45.2,  # Simulated latency
            "status": "passed",
        }
        
        logger.info(f"Endpoint test completed: {endpoint_name}")
        return json.dumps(test_result)
        
    except Exception as e:
        logger.error(f"Error testing endpoint: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


@mcp.tool()
async def get_deployment_status(deployment_id: str) -> str:
    """
    Get the current status of a deployment.
    
    Args:
        deployment_id: ID of the deployment
    
    Returns:
        JSON string with deployment status
    """
    try:
        metadata = load_deployment_metadata(deployment_id)
        if not metadata:
            return json.dumps({
                "success": False,
                "error": f"Deployment {deployment_id} not found",
            })
        
        result = {
            "success": True,
            "deployment_id": deployment_id,
            "model_id": metadata.get("model_id"),
            "status": metadata.get("status"),
            "environment": metadata.get("environment"),
            "replicas": metadata.get("replicas"),
            "health_status": metadata.get("health_status"),
            "endpoint_url": metadata.get("endpoint_url"),
            "created_at": metadata.get("created_at"),
        }
        
        if "endpoint" in metadata:
            result["endpoint_info"] = {
                "name": metadata["endpoint"].get("endpoint_name"),
                "request_count": metadata["endpoint"].get("request_count"),
                "error_count": metadata["endpoint"].get("error_count"),
                "average_latency_ms": metadata["endpoint"].get("average_latency_ms"),
            }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error getting deployment status: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


@mcp.tool()
async def list_deployments() -> str:
    """
    List all active deployments.
    
    Returns:
        JSON string with list of deployments
    """
    try:
        deployments_list = []
        for deployment_id, metadata in deployments_registry.items():
            deployments_list.append({
                "deployment_id": deployment_id,
                "model_id": metadata.get("model_id"),
                "environment": metadata.get("environment"),
                "status": metadata.get("status"),
                "health_status": metadata.get("health_status"),
                "created_at": metadata.get("created_at"),
                "endpoint_url": metadata.get("endpoint_url"),
            })
        
        result = {
            "success": True,
            "total_deployments": len(deployments_list),
            "deployments": deployments_list,
        }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error listing deployments: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


@mcp.tool()
async def rollback_deployment(deployment_id: str, previous_version: str) -> str:
    """
    Rollback a deployment to a previous version.
    
    Args:
        deployment_id: ID of the deployment to rollback
        previous_version: ID of the previous model version
    
    Returns:
        JSON string with rollback status
    """
    try:
        logger.info(f"Rolling back deployment: {deployment_id}")
        
        metadata = load_deployment_metadata(deployment_id)
        if not metadata:
            return json.dumps({
                "success": False,
                "error": f"Deployment {deployment_id} not found",
            })
        
        # Update deployment metadata
        metadata["model_id"] = previous_version
        metadata["status"] = "rolled_back"
        metadata["rolled_back_at"] = datetime.now().isoformat()
        save_deployment_metadata(deployment_id, metadata)
        
        result = {
            "success": True,
            "deployment_id": deployment_id,
            "previous_version": previous_version,
            "status": "rolled_back",
            "message": f"Deployment rolled back to version {previous_version}",
            "timestamp": datetime.now().isoformat(),
        }
        
        logger.info(f"Rollback completed: {deployment_id}")
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error rolling back deployment: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


@mcp.tool()
async def update_model(
    deployment_id: str,
    new_model_path: str,
    new_model_id: str,
) -> str:
    """
    Update a deployed model with a new version.
    
    Args:
        deployment_id: ID of the deployment to update
        new_model_path: Path to the new model file
        new_model_id: ID of the new model
    
    Returns:
        JSON string with update status
    """
    try:
        logger.info(f"Updating model in deployment: {deployment_id}")
        
        metadata = load_deployment_metadata(deployment_id)
        if not metadata:
            return json.dumps({
                "success": False,
                "error": f"Deployment {deployment_id} not found",
            })
        
        # Store previous model info for potential rollback
        previous_model_id = metadata.get("model_id")
        
        # Update deployment metadata
        metadata["model_id"] = new_model_id
        metadata["model_path"] = new_model_path
        metadata["status"] = "updated"
        metadata["updated_at"] = datetime.now().isoformat()
        metadata["previous_model_id"] = previous_model_id
        save_deployment_metadata(deployment_id, metadata)
        
        result = {
            "success": True,
            "deployment_id": deployment_id,
            "new_model_id": new_model_id,
            "previous_model_id": previous_model_id,
            "status": "updated",
            "message": f"Model updated successfully in deployment",
            "timestamp": datetime.now().isoformat(),
        }
        
        logger.info(f"Model update completed: {deployment_id}")
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error updating model: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


def main():
    """Run the MCP server."""
    logger.info("Starting ML Deployment Server...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
