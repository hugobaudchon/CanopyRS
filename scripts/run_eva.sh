#!/bin/bash

# Array of sites
sites=("ZF2" "TBS" "BCI")

# Array of models
models=("deepforest_sam2" "detectree2" "dino_sam2")

# Loop through all combinations
for site in "${sites[@]}"; do
    for model in "${models[@]}"; do
        echo "========================================"
        echo "Running evaluation: Site=${site}, Model=${model}"
        echo "========================================"
        
        python run_evalselvamask.py --site "${site}" --model "${model}"
        
        # Check if command was successful
        if [ $? -eq 0 ]; then
            echo "✓ Successfully completed: ${site} - ${model}"
        else
            echo "✗ Error running: ${site} - ${model}"
        fi
        
        echo ""
    done
done

echo "========================================"
echo "All evaluations completed!"
echo "========================================"