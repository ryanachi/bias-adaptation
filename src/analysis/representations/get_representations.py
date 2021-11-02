from datasets import load_metric
import numpy as np
import os
from quinine import Quinfig, QuinineArgumentParser, tstring, tboolean, tfloat, tinteger, stdict, stlist, default, nullable, required
import torch
from transformers import AutoModel, AutoTokenizer
from pathlib import Path 
from tqdm import tqdm 
import pickle
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from data_loader import fetch_wikipedia, fetch_w2v_GloVe_embeddings, fetch_coha_word2vec_embeddings, fetch_gpt2_data, fetch_occupation_statistics
from preprocess_contextualized_representations import fetch_canonical_embeddings, get_word2positions
cwd = Path.cwd()

# Step 1: Vocabulary
from word_list import aligned_word_lists, unaligned_word_lists, target_words
all_words = set()

for word_tuple in list(aligned_word_lists):
    for word in word_tuple:
        all_words.add(word)

for word_list in unaligned_word_lists.values():
    for word in word_list:
        all_words.add(word)
        
for word_list in target_words.values():
    for word in word_list:
        all_words.add(word)
        
sequences = fetch_wikipedia()
vocab = all_words
roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=True, truncation=True, model_max_length=512) 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_ROBERTA_LAYERS = 12

for layer in range(NUM_ROBERTA_LAYERS + 1):
    fetch_canonical_embeddings(sequences=sequences, vocab=vocab, layer=layer, model_name='roberta-base', model_path='roberta_base', file_descriptor='aug_17_21', is_model_adapted=False, device='cpu')
