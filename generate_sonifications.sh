#!/bin/bash

export CUDA_VISIBLE_DEVICES=""

M_VALUES=(1 3 5 10 20 40)
LAMBDA_VALUES=(0.1 0.25 0.5 0.75 0.9)

for m in "${M_VALUES[@]}"; do
    for lam in "${LAMBDA_VALUES[@]}"; do
        
        MODEL_FILE="saves/model_M${m}_L${lam}.pth"
        
        if [ -f "$MODEL_FILE" ]; then
            echo "--------------------------------------------------------"
            echo "Generation for M=$m | Lambda=$lam (Saving all $m prototypes)"
            
            python src/generate_prototypes.py \
                --lambda_weight $lam \
                --num_prototypes_per_class $m \
                --num_prototypes_to_save $m

        else
            echo "Model M=$m | Lambda=$lam not found. Skipping."
        fi
        
    done
done