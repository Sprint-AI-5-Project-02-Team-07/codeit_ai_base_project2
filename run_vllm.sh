#!/bin/bash
# run_vllm.sh
# Run this on GCP VM to start the vLLM server for Scenario A.

MODEL="google/gemma-2-9b-it"

echo "Starting vLLM server with model: $MODEL"
echo "Ensure you have GPU enabled and 'vllm' installed."

# Run vLLM serving
# --dtype auto: Auto-detect float16/bfloat16
# --api-key EMPTY: No key required for local access
vllm serve $MODEL --dtype auto --api-key EMPTY
