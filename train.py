"""
train.py

Core training script -- Fill in with more
repository/project-specific training details!

Run with: `python train.py --config conf/config.yaml`
"""


from datasets import load_dataset, load_metric
import numpy as np
import pandas as pd
from pathlib import Path
from quinine import Quinfig, QuinineArgumentParser, tstring, tboolean, tfloat, tinteger, stdict, stlist, default, nullable, required
from scipy import stats
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

from conf.finetuning.sentiment_analysis.schema import get_train_schema
from src.data.preprocess_dataset import load_dataset_splits
from src.adaptation.finetuning import finetune

STORAGE_FOLDER = "/u/scr/nlp/mercury/bias-adaptation"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train() -> None:
    # Parse Quinfig (via Quinine Argparse Binding)
    quinfig = QuinineArgumentParser(schema=get_train_schema()).parse_quinfig()
    
    # Datasets
    quinfig.dataset_to_load = quinfig.train_dataset
    dataset_splits = load_dataset_splits(quinfig=quinfig, dataset_cache_dir=f"{STORAGE_FOLDER}/datasets/huggingface", tokenizer_cache_dir=f"{STORAGE_FOLDER}/tokenizer") # (train, test)
    finetune(quinfig, dataset_splits)


if __name__ == "__main__":
    train()
