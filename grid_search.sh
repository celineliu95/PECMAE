#!/bin/bash

mkdir -p saves
mkdir -p logs

M_VALUES=(1 3 5 10 20 40)
LAMBDA_VALUES=(0.1 0.25 0.5 0.75 0.9)
EPOCHS=50

for m in "${M_VALUES[@]}"; do
    for lam in "${LAMBDA_VALUES[@]}"; do
        
        MODEL_NAME="model_M${m}_L${lam}"
        SAVE_PATH="saves/${MODEL_NAME}.pth"
        LOG_PATH="logs/${MODEL_NAME}.txt"

        echo "--------------------------------------------------------"
        echo "Training in progress: M=$m | Lambda=$lam"
        echo "Saving weights to: $SAVE_PATH"
        echo "Redirecting logs to: $LOG_PATH"
        
        python src/train.py \
            --num_epochs $EPOCHS \
            --lambda_weight $lam \
            --num_prototypes_per_class $m \
            --num_workers 0 \
            --save_path $SAVE_PATH > "$LOG_PATH" 2>&1
        
        echo "Finished for $MODEL_NAME"
        
    done
done

BEST_LINE=$(grep -H "Balanced Accuracy" logs/*.txt | sort -t ':' -k3 -nr | head -n 1)
BEST_FILE=$(echo "$BEST_LINE" | cut -d ':' -f 1)

BEST_M=$(echo "$BEST_FILE" | sed -E 's/.*_M([0-9]+)_L.*/\1/')
BEST_L=$(echo "$BEST_FILE" | sed -E 's/.*_L([0-9.]+)\.txt/\1/')

echo "Best hyperparameters: M=$BEST_M, Lambda=$BEST_L"