import torch
from transformers import AutoModel, AutoTokenizer
from pathlib import Path 
from tqdm import tqdm 
import pickle
cwd = Path.cwd()

from src.analysis.representations.preprocess_contextualized_representations import fetch_canonical_embeddings

STORAGE_DIR = "/u/scr/nlp/mercury/bias-adaptation/datasets/contexts/"

with open(STORAGE_DIR + 'wikipedia_2000000contexts-len=1.pkl20_sents.filter_size=857.pickle', 'rb') as pickle_file:
    content = pickle.load(pickle_file)
    
representations, word2count, missing_words = fetch_canonical_embeddings(
                                                sequences=content,
                                                layers=12,
                                                model_name='roberta-base',
                                                model_path=None, #HF website
                                                file_descriptor='aug23',
                                                is_model_adapted=False,
                                                device='cpu',
                                                sentences_per_context=1,
                                                pickled=False)