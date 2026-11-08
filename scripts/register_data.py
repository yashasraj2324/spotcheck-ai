"""Upload raw HAM10000 to the workspace's Blob datastore and register it as a versioned data asset.

Uses the azure-ai-ml SDK v2. Needs AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP and
AZURE_ML_WORKSPACE (from .env) and an `az login` session (DefaultAzureCredential uses it).

Example: python scripts/register_data.py --version 1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential

from skin_lesion.data.metadata import build_metadata


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env reader (KEY=VALUE lines) so the script needs no extra dependency."""
    env = Path(path)
    if not env.is_file():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="HAM10000 folder (default: env HAM10000_DIR)")
    parser.add_argument("--name", default="ham10000-raw")
    parser.add_argument("--version", required=True, help="asset version, e.g. 1")
    args = parser.parse_args()

    load_dotenv()
    root = Path(args.root or os.environ.get("HAM10000_DIR", "data/HAM10000"))
    build_metadata(root)  # fail fast if the metadata CSV or image folders are missing

    client = MLClient(
        DefaultAzureCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE"],
    )
    asset = Data(
        name=args.name,
        version=args.version,
        type=AssetTypes.URI_FOLDER,
        path=str(root),  # the SDK uploads this folder to the default blob datastore
        description="HAM10000 raw metadata CSV and image folders (research use).",
        tags={"project": "skin-lesion"},
    )
    registered = client.data.create_or_update(asset)
    print(f"Registered data asset {registered.name}:{registered.version}")
    print("Set in .env:")
    print(f"  DATASET_ASSET_NAME={registered.name}")
    print(f"  DATASET_ASSET_VERSION={registered.version}")


if __name__ == "__main__":
    main()
