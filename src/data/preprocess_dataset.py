from datasets import load_dataset, load_metric
import math
import numpy as np
import os
import pandas as pd
from quinine import Quinfig
import torch
from transformers import AutoTokenizer
import random

EEC_FOLDER = "/u/scr/nlp/mercury/bias-adaptation/datasets/EEC"

def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)

def load_dataset_splits(quinfig, dataset_cache_dir, tokenizer_cache_dir):
    # Reproducibility
    random.seed(quinfig.seed)
    np.random.seed(quinfig.seed)
    torch.manual_seed(quinfig.seed)

    # Datasets
            
    if quinfig.dataset_to_load == "eec":
        male_dataset = torch.load(f"{EEC_FOLDER}/male_eec_dataset.pt")
        female_dataset = torch.load(f"{EEC_FOLDER}/female_eec_dataset.pt")
        return (male_dataset, female_dataset)
    
    elif quinfig.dataset_to_load == 'sst2':
        raw_datasets = load_dataset(path='glue', name='sst2', cache_dir=dataset_cache_dir)        
    else:
        raw_datasets = load_dataset(quinfig.dataset_to_load, cache_dir=dataset_cache_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(quinfig.model_name, cache_dir=tokenizer_cache_dir)
    
    def tokenize_function(examples):
        if quinfig.dataset_to_load == 'sst2':
            return tokenizer(examples["sentence"], padding="max_length", truncation=True)
        elif quinfig.dataset_to_load == 'amazon_polarity':
            return tokenizer(examples["content"], padding="max_length", truncation=True)
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets["train"].shuffle()
    tokenized_datasets["test"].shuffle()
    
    divisor = lcm(quinfig.num_save_checkpoints, quinfig.evals_per_epoch) * quinfig.per_device_train_batch_size
    
    if quinfig.dataset_to_load == "sst2":
        tokenized_datasets["validation"].shuffle()
        train_dataset = tokenized_datasets["train"]
        test_dataset = tokenized_datasets["validation"]
        train_dataset = train_dataset.select(range(len(train_dataset) - len(train_dataset) % divisor))
        if quinfig.dataset_to_load == quinfig.train_dataset:
            quinfig.train_set_size = len(train_dataset)
        return (train_dataset, test_dataset)
            
    else:
        train_dataset = tokenized_datasets["train"]
        train_dataset = train_dataset.select(range(len(train_dataset) - len(train_dataset) % divisor))
        test_dataset = tokenized_datasets["test"]
           
    if quinfig.dataset_to_load == quinfig.train_dataset:
        quinfig.train_set_size = len(train_dataset)
    return (train_dataset, test_dataset)