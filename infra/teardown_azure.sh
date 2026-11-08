#!/usr/bin/env bash
# Delete everything created by setup_azure.sh by removing its resource group.
# Refuses to touch any group that lacks the project=skin-lesion tag, so it cannot hit your other projects.
set -euo pipefail

cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi

RG="${AZURE_RESOURCE_GROUP:-rg-skin-lesion}"
WS="${AZURE_ML_WORKSPACE:-mlw-skin-lesion}"

TAGGED="$(az group show -n "$RG" --query "tags.project" -o tsv 2>/dev/null || true)"
if [ "$TAGGED" != "skin-lesion" ]; then
  echo "Refusing: resource group '$RG' is missing or lacks the tag project=skin-lesion." >&2
  exit 1
fi

echo "This will permanently delete resource group '$RG' and everything in it:"
az resource list -g "$RG" --query "[].{name:name,type:type}" -o table
read -r -p "Type the resource group name to confirm: " CONFIRM
[ "$CONFIRM" = "$RG" ] || { echo "Aborted."; exit 1; }

# Azure ML workspaces are soft-deleted by default (name stays reserved); purge it first.
if az ml workspace show -n "$WS" -g "$RG" -o none 2>/dev/null; then
  az ml workspace delete -n "$WS" -g "$RG" --permanently-delete --all-resources -y
fi
az group delete -n "$RG" -y
echo "Deleted '$RG'."
