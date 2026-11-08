#!/usr/bin/env bash
# Idempotent Azure setup: resource group, storage account, Azure ML workspace (and optional GPU compute).
# Safe to re-run; each step checks whether the resource exists first. Reads settings from .env / env vars.
#
#   bash infra/setup_azure.sh            # core resources only (no compute, no running cost)
#   WITH_GPU_COMPUTE=yes bash infra/setup_azure.sh   # also a GPU cluster, min nodes 0 (needs GPU quota)
set -euo pipefail

cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi   # load settings; .env is gitignored

LOCATION="${AZURE_LOCATION:-centralindia}"
RG="${AZURE_RESOURCE_GROUP:-rg-skin-lesion}"
WS="${AZURE_ML_WORKSPACE:-mlw-skin-lesion}"
TAG="project=skin-lesion"   # teardown refuses to delete a resource group without this tag

SUB_ID="$(az account show --query id -o tsv)"
# Storage names are global, 3-24 chars, lowercase alphanumeric: derive a stable suffix from the subscription.
SUFFIX="$(printf '%s' "$SUB_ID" | md5sum | cut -c1-8)"
SA="${AZURE_STORAGE_ACCOUNT:-stskinlesion${SUFFIX}}"

echo "Subscription : $SUB_ID"
echo "Location     : $LOCATION"
echo "Resource grp : $RG"
echo "Storage acct : $SA"
echo "ML workspace : $WS"

# 1. Resource group: the container for everything, so one delete removes it all.
az group create -n "$RG" -l "$LOCATION" --tags "$TAG" -o none

# 2. Storage account (Blob): LRS = cheapest redundancy; no public blob access; TLS 1.2 minimum.
if ! az storage account show -n "$SA" -g "$RG" -o none 2>/dev/null; then
  az storage account create -n "$SA" -g "$RG" -l "$LOCATION" \
    --sku Standard_LRS --kind StorageV2 --min-tls-version TLS1_2 \
    --allow-blob-public-access false --tags "$TAG" -o none
fi
SA_ID="$(az storage account show -n "$SA" -g "$RG" --query id -o tsv)"

# 3. Azure ML workspace, reusing the storage account above as its default datastore.
#    The workspace also auto-creates a Key Vault and an Application Insights instance.
if ! az ml workspace show -n "$WS" -g "$RG" -o none 2>/dev/null; then
  az ml workspace create -n "$WS" -g "$RG" -l "$LOCATION" --storage-account "$SA_ID" --tags "$TAG" -o none
fi

# 4. Optional GPU cluster. min-instances 0 = no nodes (no cost) when idle; idle timeout scales back to 0.
if [ "${WITH_GPU_COMPUTE:-no}" = "yes" ]; then
  SIZE="${GPU_VM_SIZE:?set GPU_VM_SIZE, e.g. Standard_NC4as_T4_v3 (check quota first)}"
  if ! az ml compute show -n gpu-cluster -w "$WS" -g "$RG" -o none 2>/dev/null; then
    az ml compute create -n gpu-cluster --type amlcompute --size "$SIZE" \
      --min-instances 0 --max-instances 1 --idle-time-before-scale-down 120 -w "$WS" -g "$RG" -o none
  fi
fi

# 5. Budget alert (best effort: many student/sponsored subscriptions do not support the Budgets API).
BUDGET_USD="${BUDGET_USD:-20}"
if [ -n "${ALERT_EMAIL:-}" ]; then
  az consumption budget create --budget-name skin-lesion-budget --amount "$BUDGET_USD" --category cost \
    --time-grain monthly --start-date "$(date +%Y-%m-01)" --end-date "$(date -d '+1 year' +%Y-%m-01)" \
    --resource-group "$RG" --notifications \
    "{\"actual80\":{\"enabled\":true,\"operator\":\"GreaterThan\",\"threshold\":80,\"contactEmails\":[\"$ALERT_EMAIL\"]}}" \
    -o none 2>/dev/null \
    && echo "Budget alert created ($BUDGET_USD USD)." \
    || echo "Budget API not available for this subscription: set an alert in the portal (Cost Management > Budgets)."
else
  echo "Set ALERT_EMAIL to try creating a budget alert; otherwise set one in the portal (Cost Management > Budgets)."
fi

echo
echo "Done. Add to .env (never commit it):"
echo "  AZURE_SUBSCRIPTION_ID=$SUB_ID"
echo "  AZURE_RESOURCE_GROUP=$RG"
echo "  AZURE_ML_WORKSPACE=$WS"
echo "  MLFLOW_TRACKING_URI=$(az ml workspace show -n "$WS" -g "$RG" --query mlflow_tracking_uri -o tsv)"
