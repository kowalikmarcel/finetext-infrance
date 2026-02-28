source ./classification_config
NUM_WORKERS=12
NETWORK_VOLUME_ID="ob4cd0ovyx"
for i in $(seq 0 $(( $NUM_WORKERS - 1 ))); 
do
    runpodctl create pod \
    --secureCloud \
    --name "fineweb_classify_${i}" \
    --gpuType "NVIDIA GeForce RTX 4090" \
    --templateId "runpod-torch-v240" \
    --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
    --containerDiskSize 50 \
    --vcpu 6 \
    --mem 35 \
    --networkVolumeId "${NETWORK_VOLUME_ID}" \
    --volumePath "${PERSISTENT_STORAGE}" \
    --env POD_ID="${i}" \
    --env NUM_WORKERS="${NUM_WORKERS}" \
    --args "bash -c 'git clone https://github.com/kowalikmarcel/finetext-infrance.git && cd finetext-infrance && ./startup.sh'" \
    --ports "22/tcp"

    sleep 2
done
