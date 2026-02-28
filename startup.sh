#/bin/bash

set -a
source ./classification_config
set +a

python -m pip install -r requirements.txt

LOG_FILE="${PERSISTENT_STORAGE}/fineweb2_bis/data_labeling/logs/worker_${POD_ID}.txt"
mkdir -p "${PERSISTENT_STORAGE}/logs"
exec > >(tee -a "$LOG_FILE") 2>&1

cp -a "${PERSISTENT_STORAGE}/${MODEL_CHECKPOINT}" .


python classify.py
if [ -n "$RUNPOD_POD_ID" ]; then 
    runpodctl remove pod $RUNPOD_POD_ID
fi
