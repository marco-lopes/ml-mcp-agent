#!/usr/bin/env python3
"""
Simple example of downloading datasets from Kaggle using kagglehub

This demonstrates the corrected kagglehub integration.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load Kaggle credentials from .env file
load_dotenv()
if not os.getenv("KAGGLE_KEY"):
    print("Error: KAGGLE_KEY not found. Set it in .env file or environment.")
    sys.exit(1)

print("=" * 70)
print("  KAGGLE DATASET DOWNLOAD EXAMPLE")
print("=" * 70)

# Import kagglehub
try:
    import kagglehub
    print("\n✅ kagglehub library imported successfully")
except ImportError:
    print("\n❌ kagglehub not installed")
    print("Install with: pip install kagglehub")
    exit(1)

# Check token
if "KAGGLE_KEY" in os.environ:
    token = os.environ["KAGGLE_KEY"]
    print(f"✅ Token configured: {token[:20]}...")
else:
    print("❌ KAGGLE_KEY not set")
    exit(1)

print("\n" + "=" * 70)
print("  DOWNLOADING IRIS DATASET")
print("=" * 70 + "\n")

try:
    # Download dataset using kagglehub
    dataset_ref = "uciml/iris"
    print(f"📥 Downloading: {dataset_ref}")
    print("   (This will use kagglehub's cache directory)\n")
    
    # kagglehub.dataset_download returns the path to cached files
    download_path = kagglehub.dataset_download(dataset_ref)
    
    print(f"\n✅ Download successful!")
    print(f"📁 Dataset location: {download_path}")
    
    # List files in the dataset
    dataset_path = Path(download_path)
    
    if dataset_path.exists():
        print(f"\n📄 Files in dataset:")
        
        total_size = 0
        file_count = 0
        
        for file in sorted(dataset_path.rglob("*")):
            if file.is_file():
                size_bytes = file.stat().st_size
                size_kb = size_bytes / 1024
                total_size += size_bytes
                file_count += 1
                
                print(f"  {file_count}. {file.name}")
                print(f"     Size: {size_kb:.1f} KB")
                print(f"     Path: {file}")
        
        print(f"\n📊 Summary:")
        print(f"   Total files: {file_count}")
        print(f"   Total size: {total_size / 1024:.1f} KB")
        
        # Show first few lines of CSV if exists
        csv_files = list(dataset_path.glob("*.csv"))
        if csv_files:
            csv_file = csv_files[0]
            print(f"\n📝 Preview of {csv_file.name}:")
            print("   " + "-" * 60)
            
            with open(csv_file, 'r') as f:
                for i, line in enumerate(f):
                    if i < 5:  # Show first 5 lines
                        print(f"   {line.rstrip()}")
                    else:
                        break
            print("   " + "-" * 60)
    
    print("\n" + "=" * 70)
    print("  USAGE IN ML PIPELINE")
    print("=" * 70 + "\n")
    
    print("You can now use this dataset for training:")
    print(f"""
from main import MLAgent

agent = MLAgent()
agent.start()

# The dataset is already downloaded at:
dataset_path = "{download_path}/Iris.csv"

# Train a model
result = agent.train_model(
    model_name="iris_classifier",
    model_type="random_forest",
    dataset_path=dataset_path,
    target_column="variety",  # or "species"
    task_type="classification",
    hyperparameters={{"n_estimators": 100}}
)

print(f"Accuracy: {{result['test_metrics']['val_accuracy']}}")
agent.stop()
""")
    
    print("=" * 70)
    print("  SUCCESS!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
