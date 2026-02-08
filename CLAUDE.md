# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML Orchestration Agent that integrates Machine Learning pipelines with the Model Context Protocol (MCP). It automates training, validating, and deploying ML models using Kaggle datasets through a distributed architecture of specialized MCP servers communicating via JSON-RPC over stdio.

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r servers/training_server/requirements.txt
pip install -r servers/deployment_server/requirements.txt
pip install -r servers/kaggle_server/requirements.txt

# Run examples
python examples/standalone_training_example.py    # Direct training, no MCP
python examples/real_training_example.py           # Full MCP pipeline
python examples/train_and_deploy.py                # Train + deploy flow
python examples/kaggle_download_example.py         # Dataset download only
```

Kaggle credentials must be configured via `KAGGLE_KEY` env var or `.env` file — never hardcode tokens in source files.

There is no formal test suite, build system, or linter configured. The `tests/` directory exists but is empty.

## Architecture

The system follows a **distributed MCP pattern** with three layers:

### Agent Layer (`agent/`)
- **`main.py`** — `MLAgent` class, top-level entry point wrapping the orchestrator
- **`orchestrator.py`** — `MLPipelineOrchestrator`, coordinates the 6-step pipeline: train → validate → save → deploy → create endpoint → test endpoint. Manages lifecycle of all three MCP server clients.
- **`mcp_client.py`** — `MCPClient` base class that spawns server subprocesses and communicates via JSON-RPC. Specialized subclasses: `TrainingClient`, `DeploymentClient`, `KaggleClient`.

### Server Layer (`servers/`)
Each server is a standalone FastMCP process exposing tools via `@mcp.tool()` decorators:

- **`training_server/`** — Real scikit-learn model training (classification + regression). `main.py` exposes MCP tools (train_model, validate_model, predict, etc.). `models.py` contains the `MLModel` wrapper with scaling, encoding, and metric calculation. `utils.py` has cross-validation and data preprocessing helpers.
- **`deployment_server/`** — Model deployment management with simulated endpoints. Tracks deployments, supports rollback and update operations.
- **`kaggle_server/`** — Dataset search/download via `kagglehub` (downloads) and `kaggle` CLI (search). Supports OAuth (`KAGGLE_KEY` env var) and traditional `~/.kaggle/kaggle.json` credentials.

### Communication Flow
```
Agent (orchestrator) → spawns subprocess → MCP Server (FastMCP)
                     ← JSON-RPC over stdio ←
```

### Data Persistence
- Models: `/tmp/ml_models/` (joblib + JSON metadata)
- Deployments: `/tmp/ml_deployments/` (JSON metadata)
- Datasets: `~/.cache/kagglehub/datasets/`

## Key Patterns

- All MCP tool functions are `async` (FastMCP requirement) even though current I/O is synchronous
- Servers maintain both in-memory registries and disk-persisted JSON metadata; `active_models` is capped at 50 entries with automatic eviction of the oldest
- The orchestrator stops the pipeline early if model accuracy falls below a configurable `min_accuracy` threshold
- Kaggle credentials can be set via `KAGGLE_KEY` env var, `.env` file, or `~/.kaggle/kaggle.json`
- Supported sklearn models are registered in `CLASSIFICATION_MODELS` and `REGRESSION_MODELS` dicts in `training_server/models.py`

## Common Pitfalls

When modifying this codebase, watch out for these issues that were previously found and fixed:

- **Metrics key alignment**: The training server validation response uses `"metrics"` as the key. The orchestrator reads `validation_result.get("metrics", {}).get("accuracy", 0)`. Changing either side without the other breaks the accuracy threshold check.
- **target_column propagation**: `validate_model()` requires `target_column` as a parameter. It must be passed through the entire chain: `MLAgent.execute_pipeline()` → `orchestrator.train_and_deploy()` → `TrainingClient.validate_model()` → training server.
- **Model file extension**: Models are saved as `.joblib` (via `models.py`). Any path construction in the orchestrator or elsewhere must use `.joblib`, not `.pkl`.
- **Label encoding in validate()**: `MLModel.validate()` encodes `y_val` and uses raw `model.predict()` (which returns encoded values). Do not call `self.predict()` here — that method applies `inverse_transform`, causing a double-encoding mismatch.
- **Kaggle sort_by validation**: The `search_datasets` tool validates `sort_by` against `ALLOWED_SORT_OPTIONS` (`hottest`, `votes`, `updated`, `active`). New sort options must be added to this whitelist.
- **Path traversal protection**: `download_dataset` restricts `output_dir` to paths under the user's home directory or `/tmp`.
- **MCP client error handling**: `call_tool()` catches `BrokenPipeError`/`IOError` on stdin/stdout and `JSONDecodeError` on responses. If the server crashes, the client returns an error dict instead of raising.
