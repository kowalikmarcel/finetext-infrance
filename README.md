# finetext-infrance — Distributed GPU Inference

This submodule contains the inference system used to score the full ~160M document Polish corpus with the fine-tuned educational quality classifier. It orchestrates a fleet of RunPod RTX 4090 pods, each processing an independent shard of the pre-tokenized dataset in parallel.

This is the "GPU step" that sits between `data_processing/tokenize_for_roberta.py` (pre-tokenization) and `data_processing/add_predictions_to_datasets.py` (score merging) in the main pipeline.

## Execution Order

```
1. [prerequisite] Pre-tokenized datasets must exist on the network volume
   (produced by data_processing/tokenize_for_roberta.py)

2. Edit classification_config         — set dataset paths, model checkpoint, batch size

3. spinup_pods_runpod.sh              — provision NUM_WORKERS RTX 4090 pods on RunPod
   (each pod auto-runs startup.sh on boot)

4. startup.sh                         — [runs on each pod] install deps, copy model, run classify.py
   classify.py                        — [runs on each pod] score one shard, write predictions to parquet

5. [after all pods finish] predictions are available in PREDICTION_DIR on the network volume
   (consumed by data_processing/add_predictions_to_datasets.py)
```

Pods self-terminate after inference completes.

---

## Files

### `classify.py`

Core inference script. All configuration is read from environment variables (set by `classification_config`).

**Sharding:** each worker takes `dataset.shard(num_shards=NUM_WORKERS, index=POD_ID)` so the 12 pods divide the data evenly without communication.

**Throughput optimizations:**
- Sorts the shard by sequence length (descending) before batching — minimizes padding waste
- fp16 model (`torch_dtype=torch.float16`)
- Flash SDP and memory-efficient SDP enabled
- `pin_memory=True`, `persistent_workers=True` on the DataLoader
- `torch.inference_mode()` throughout

**Output:** for each dataset in `DATASETS`, writes a Parquet file to `PREDICTION_DIR`:
```
preds/<dataset_name>_part_<POD_ID>.parquet
```
Each file contains two columns: `id` (document identifier) and `prediction` (scalar score).

