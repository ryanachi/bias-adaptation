from datasets import load_dataset, load_metric
import numpy as np
import pandas as pd
from pathlib import Path
from quinine import Quinfig, QuinineArgumentParser, tstring, tboolean, tfloat, tinteger, stdict, stlist, default, nullable, required
from scipy import stats
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

from conf.evaluation.sentiment_analysis.schema import get_eval_schema
from src.data.preprocess_dataset import load_dataset_splits
from src.data.download_eec import EECDataset
from src.analysis.evaluate_dataset import analyze_bias

STORAGE_FOLDER = "/u/scr/nlp/mercury/bias-adaptation"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_bias() -> None:
    # Parse Quinfig (via Quinine Argparse Binding)
    quinfig = QuinineArgumentParser(schema=get_eval_schema()).parse_quinfig()
    
    if quinfig.train_dataset == "yelp_polarity":
#         quinfig.max_checkpoint_step = 210_000
        save_steps = 21_000
    elif quinfig.train_dataset == "sst2":
#         quinfig.max_checkpoint_step = 25_200
        save_steps = 2_520
    elif quinfig.train_dataset == "imdb":
#         quinfig.max_checkpoint_step = 25_000
        save_steps = 2_500
        
    for dataset in quinfig.evaluation_datasets: #just EEC for now
        for run_id in range(1, 6):
            for checkpoint in range(1, 11):  
                quinfig.max_checkpoint_step = checkpoint * save_steps
                quinfig.run_id = f"{quinfig.adaptation_process}-seed-{run_id}"
                quinfig.dataset_to_load = dataset
            # Datasets
                dataset_splits = load_dataset_splits(quinfig=quinfig, dataset_cache_dir=f"{STORAGE_FOLDER}/datasets/huggingface", tokenizer_cache_dir=f"{STORAGE_FOLDER}/tokenizer") # (male, female)
                analyze_bias(quinfig, dataset_splits)

if __name__ == "__main__":
    evaluate_bias()