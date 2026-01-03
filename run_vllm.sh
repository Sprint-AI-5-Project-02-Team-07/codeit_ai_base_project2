#!/bin/bash
# run_vllm.sh
# Run this on GCP VM to start the vLLM server for Scenario A.

MODEL=$(uv run python -c "import yaml; print(yaml.safe_load(open('config/config.yaml'))['model']['local_llm'])")

echo "Starting vLLM server with model: $MODEL"
echo "Ensure you have GPU enabled and 'vllm' installed."

# Run vLLM serving
# --dtype auto: Auto-detect float16/bfloat16
# --api-key EMPTY: No key required for local access
uv run vllm serve $MODEL --dtype auto --api-key EMPTY --port 32000