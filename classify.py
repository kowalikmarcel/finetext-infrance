import datasets
import torch
import numpy as np
import time
import os
import gc
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from tqdm_loggable.auto import tqdm
import logging
from datasets import disable_progress_bar
disable_progress_bar()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
)

DATASETS_DIRS = os.getenv('DATASETS').split(',')
CHECKPOINT_PATH = os.getenv('MODEL_CHECKPOINT')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32)) 
PREDS_DIR = os.getenv('PREDICTION_DIR')
POD_ID = int(os.getenv('POD_ID', 0))
NUM_WORKERS = int(os.getenv('NUM_WORKERS', 1))

device = "cuda"


torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False) 



tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT_PATH,
    num_labels=1,
    problem_type="regression",
    torch_dtype=torch.float16, 
    device_map="cuda"
)
model.eval()
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
logging.info('Model, tokenizer loaded.')

for dir_path in DATASETS_DIRS:
    dataset_name = os.path.basename(dir_path.rstrip('/'))
    logging.info(f"--- Processing Dataset: {dataset_name} ---")

    dataset = datasets.load_from_disk(dir_path)
    dataset = dataset.shard(num_shards=NUM_WORKERS, index=POD_ID)
    logging.info(f"Worker {POD_ID} taking {len(dataset)} samples from {dataset_name}")

    dataset = dataset.map(
        lambda x: {"length": [len(i) for i in x["input_ids"]]}, 
        batched=True, 
        batch_size=10000, 
        num_proc=os.cpu_count()
    )
    
    dataset = dataset.sort("length", reverse=True) 
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    all_preds = []
    
    with torch.inference_mode():
        start = time.time()
        for batch in tqdm(dataloader, desc=f"Inf: {dataset_name}-{POD_ID}", mininterval=90):
            inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k in tokenizer.model_input_names}
            outputs = model(**inputs)
            preds = outputs.logits.squeeze(-1).float().cpu().numpy()
            all_preds.append(preds)
            
        if device == "cuda":
            torch.cuda.synchronize()    
        total = time.time() - start

    logging.info(f"Finished {dataset_name} in {total:.2f}s ({len(dataset)/total:.2f} samples/s)")

    predictions = np.concatenate(all_preds)
    del all_preds
    
    dataset.set_format(type=None, columns=['id'])
    dataset = dataset.add_column("prediction", predictions)
    
    os.makedirs(PREDS_DIR, exist_ok=True)
    
    save_path = f'{PREDS_DIR}/{dataset_name}_part_{POD_ID}.parquet'
    dataset.select_columns(["id", "prediction"]).to_parquet(save_path)

    del dataset
    del predictions
    gc.collect() 
    torch.cuda.empty_cache() 

logging.info("All datasets processed.")