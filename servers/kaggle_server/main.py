#!/usr/bin/env python3
"""
MCP Server for Kaggle Dataset Management

This server provides tools for searching and downloading datasets from Kaggle.
It communicates via the Model Context Protocol using JSON-RPC.
Uses kagglehub library for better integration with Kaggle.
"""

import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("kaggle-dataset-server")

# Storage for dataset metadata
DATASETS_DIR = Path("/tmp/kaggle_datasets")
DATASETS_DIR.mkdir(exist_ok=True)

# Registry for downloaded datasets
datasets_registry: dict[str, dict[str, Any]] = {}


def check_kagglehub_available() -> bool:
    """Check if kagglehub is installed."""
    try:
        import kagglehub
        return True
    except ImportError:
        return False


def check_kaggle_credentials() -> bool:
    """Check if Kaggle credentials are configured."""
    # Check for KAGGLE_KEY environment variable (OAuth token)
    if "KAGGLE_KEY" in os.environ:
        return True
    
    # Check for kaggle.json file
    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_config.exists():
        return True
    
    # Check for KAGGLE_USERNAME and KAGGLE_KEY
    if "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ:
        return True
    
    return False


def run_kaggle_command(command: list[str]) -> tuple[bool, str]:
    """
    Run a kaggle CLI command for operations not supported by kagglehub.
    
    Args:
        command: List of command arguments
    
    Returns:
        Tuple of (success, output)
    """
    try:
        result = subprocess.run(
            ["kaggle"] + command,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except FileNotFoundError:
        return False, "Kaggle CLI not installed. Install with: pip install kaggle"
    except Exception as e:
        return False, str(e)


def save_dataset_metadata(dataset_id: str, metadata: dict[str, Any]) -> None:
    """Save dataset metadata to registry."""
    datasets_registry[dataset_id] = metadata
    metadata_file = DATASETS_DIR / f"{dataset_id.replace('/', '_')}_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Dataset metadata saved: {dataset_id}")


@mcp.tool()
async def search_datasets(
    query: str,
    max_results: int = 10,
    sort_by: str = "hottest",
) -> str:
    """
    Search for datasets on Kaggle.
    
    Args:
        query: Search query term
        max_results: Maximum number of results to return (default: 10)
        sort_by: Sort order - 'hottest', 'votes', 'updated', 'active' (default: 'hottest')
    
    Returns:
        JSON string with list of datasets
    """
    try:
        logger.info(f"Searching datasets with query: {query}")
        
        if not check_kaggle_credentials():
            return json.dumps({
                "success": False,
                "error": "Kaggle credentials not configured. Please set KAGGLE_KEY environment variable or configure ~/.kaggle/kaggle.json",
            })
        
        # Use kaggle CLI for search (kagglehub doesn't have search functionality)
        command = ["datasets", "list", "-s", query, "--sort-by", sort_by, "--max-size", str(max_results)]
        success, output = run_kaggle_command(command)
        
        if not success:
            return json.dumps({
                "success": False,
                "error": f"Failed to search datasets: {output}",
            })
        
        # Parse the output (CSV format)
        lines = output.strip().split('\n')
        if len(lines) < 2:
            return json.dumps({
                "success": True,
                "total_results": 0,
                "datasets": [],
                "message": "No datasets found",
            })
        
        # Skip header line
        datasets = []
        for line in lines[1:]:
            parts = line.split(',')
            if len(parts) >= 5:
                dataset_info = {
                    "ref": parts[0].strip(),
                    "title": parts[1].strip(),
                    "size": parts[2].strip(),
                    "last_updated": parts[3].strip(),
                    "download_count": parts[4].strip() if len(parts) > 4 else "N/A",
                    "vote_count": parts[5].strip() if len(parts) > 5 else "N/A",
                }
                datasets.append(dataset_info)
        
        result = {
            "success": True,
            "query": query,
            "total_results": len(datasets),
            "datasets": datasets,
            "sort_by": sort_by,
        }
        
        logger.info(f"Found {len(datasets)} datasets")
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error searching datasets: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


@mcp.tool()
async def download_dataset(
    dataset_ref: str,
    output_dir: Optional[str] = None,
    force_download: bool = False,
) -> str:
    """
    Download a dataset from Kaggle using kagglehub.
    
    Args:
        dataset_ref: Dataset reference (format: 'username/dataset-name')
        output_dir: Output directory (default: uses kagglehub cache)
        force_download: Force download even if already cached (default: False)
    
    Returns:
        JSON string with download status and path
    """
    try:
        logger.info(f"Downloading dataset: {dataset_ref}")
        
        if not check_kagglehub_available():
            return json.dumps({
                "success": False,
                "error": "kagglehub library not installed. Install with: pip install kagglehub",
            })
        
        if not check_kaggle_credentials():
            return json.dumps({
                "success": False,
                "error": "Kaggle credentials not configured. Please set KAGGLE_KEY environment variable.",
            })
        
        import kagglehub
        
        # Download using kagglehub
        # Note: kagglehub.dataset_download returns the path where files are cached
        # It doesn't accept output_dir parameter - it uses its own cache directory
        download_path = kagglehub.dataset_download(dataset_ref)
        
        # If user specified output_dir, copy files there
        if output_dir:
            import shutil
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Copy downloaded files to specified directory
            source_path = Path(download_path)
            if source_path.is_dir():
                # Copy all files from cache to output_dir
                for item in source_path.rglob("*"):
                    if item.is_file():
                        rel_path = item.relative_to(source_path)
                        dest_file = output_path / rel_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, dest_file)
                download_path = str(output_path)
            else:
                # Single file
                shutil.copy2(source_path, output_path / source_path.name)
                download_path = str(output_path / source_path.name)
        
        # Convert to Path object
        output_path = Path(download_path)
        
        # List downloaded files
        downloaded_files = []
        if output_path.is_dir():
            for file_path in output_path.rglob("*"):
                if file_path.is_file():
                    downloaded_files.append({
                        "name": file_path.name,
                        "path": str(file_path),
                        "size_bytes": file_path.stat().st_size,
                        "relative_path": str(file_path.relative_to(output_path)),
                    })
        elif output_path.is_file():
            downloaded_files.append({
                "name": output_path.name,
                "path": str(output_path),
                "size_bytes": output_path.stat().st_size,
                "relative_path": output_path.name,
            })
        
        # Save metadata
        metadata = {
            "dataset_ref": dataset_ref,
            "output_dir": str(output_path),
            "downloaded_at": datetime.now().isoformat(),
            "files": downloaded_files,
            "total_files": len(downloaded_files),
            "method": "kagglehub",
        }
        
        save_dataset_metadata(dataset_ref, metadata)
        
        result = {
            "success": True,
            "dataset_ref": dataset_ref,
            "output_dir": str(output_path),
            "files": downloaded_files,
            "total_files": len(downloaded_files),
            "message": f"Dataset downloaded successfully to {output_path}",
        }
        
        logger.info(f"Dataset downloaded: {dataset_ref} -> {output_path}")
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


@mcp.tool()
async def list_dataset_files(dataset_ref: str) -> str:
    """
    List files in a Kaggle dataset without downloading.
    
    Args:
        dataset_ref: Dataset reference (format: 'username/dataset-name')
    
    Returns:
        JSON string with list of files
    """
    try:
        logger.info(f"Listing files for dataset: {dataset_ref}")
        
        if not check_kaggle_credentials():
            return json.dumps({
                "success": False,
                "error": "Kaggle credentials not configured.",
            })
        
        command = ["datasets", "files", "-d", dataset_ref]
        success, output = run_kaggle_command(command)
        
        if not success:
            return json.dumps({
                "success": False,
                "error": f"Failed to list dataset files: {output}",
            })
        
        # Parse output
        lines = output.strip().split('\n')
        files = []
        
        if len(lines) > 1:
            for line in lines[1:]:  # Skip header
                parts = line.split(',')
                if len(parts) >= 2:
                    files.append({
                        "name": parts[0].strip(),
                        "size": parts[1].strip(),
                    })
        
        result = {
            "success": True,
            "dataset_ref": dataset_ref,
            "total_files": len(files),
            "files": files,
        }
        
        logger.info(f"Found {len(files)} files in dataset")
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error listing dataset files: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


@mcp.tool()
async def get_dataset_metadata(dataset_ref: str) -> str:
    """
    Get metadata for a Kaggle dataset.
    
    Args:
        dataset_ref: Dataset reference (format: 'username/dataset-name')
    
    Returns:
        JSON string with dataset metadata
    """
    try:
        logger.info(f"Getting metadata for dataset: {dataset_ref}")
        
        if not check_kaggle_credentials():
            return json.dumps({
                "success": False,
                "error": "Kaggle credentials not configured.",
            })
        
        command = ["datasets", "metadata", "-d", dataset_ref]
        success, output = run_kaggle_command(command)
        
        if not success:
            return json.dumps({
                "success": False,
                "error": f"Failed to get dataset metadata: {output}",
            })
        
        # Parse JSON output
        try:
            metadata = json.loads(output)
            result = {
                "success": True,
                "dataset_ref": dataset_ref,
                "metadata": metadata,
            }
        except json.JSONDecodeError:
            result = {
                "success": True,
                "dataset_ref": dataset_ref,
                "metadata": {"raw_output": output},
            }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error getting dataset metadata: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


@mcp.tool()
async def list_downloaded_datasets() -> str:
    """
    List all datasets that have been downloaded.
    
    Returns:
        JSON string with list of downloaded datasets
    """
    try:
        downloaded_datasets = []
        
        for dataset_id, metadata in datasets_registry.items():
            downloaded_datasets.append({
                "dataset_ref": dataset_id,
                "output_dir": metadata.get("output_dir"),
                "downloaded_at": metadata.get("downloaded_at"),
                "total_files": metadata.get("total_files"),
                "method": metadata.get("method", "unknown"),
            })
        
        result = {
            "success": True,
            "total_datasets": len(downloaded_datasets),
            "datasets": downloaded_datasets,
        }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error listing downloaded datasets: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


@mcp.tool()
async def get_downloaded_dataset_path(dataset_ref: str) -> str:
    """
    Get the local path of a downloaded dataset.
    
    Args:
        dataset_ref: Dataset reference (format: 'username/dataset-name')
    
    Returns:
        JSON string with dataset path
    """
    try:
        if dataset_ref in datasets_registry:
            metadata = datasets_registry[dataset_ref]
            result = {
                "success": True,
                "dataset_ref": dataset_ref,
                "output_dir": metadata.get("output_dir"),
                "files": metadata.get("files"),
                "downloaded_at": metadata.get("downloaded_at"),
            }
        else:
            result = {
                "success": False,
                "error": f"Dataset {dataset_ref} not found in downloaded datasets",
            }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error getting dataset path: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


@mcp.tool()
async def configure_credentials(kaggle_key: str, kaggle_username: Optional[str] = None) -> str:
    """
    Configure Kaggle credentials (OAuth token or username/key pair).
    
    Args:
        kaggle_key: Kaggle API key or OAuth token (starts with KAGGLE_KEY or kagat_)
        kaggle_username: Kaggle username (optional, only needed for API key authentication)
    
    Returns:
        JSON string with configuration status
    """
    try:
        logger.info("Configuring Kaggle credentials")
        
        # Set environment variables
        os.environ["KAGGLE_KEY"] = kaggle_key
        
        if kaggle_username:
            os.environ["KAGGLE_USERNAME"] = kaggle_username
        
        # Verify credentials work
        if check_kaggle_credentials():
            result = {
                "success": True,
                "message": "Kaggle credentials configured successfully",
                "auth_type": "oauth_token" if kaggle_key.startswith(("KAGGLE_KEY", "kagat_")) else "api_key",
            }
        else:
            result = {
                "success": False,
                "error": "Failed to configure credentials",
            }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error configuring credentials: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


def main():
    """Run the MCP server."""
    logger.info("Starting Kaggle Dataset Server (kagglehub version)...")
    
    # Check if kagglehub is available
    if not check_kagglehub_available():
        logger.warning("kagglehub library not installed!")
        logger.warning("Install with: pip install kagglehub")
    
    # Check credentials on startup
    if not check_kaggle_credentials():
        logger.warning("Kaggle credentials not configured!")
        logger.warning("Set KAGGLE_KEY environment variable or configure ~/.kaggle/kaggle.json")
    else:
        logger.info("Kaggle credentials found")
    
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
