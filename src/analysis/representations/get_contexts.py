from datasets import load_metric
import numpy as np
import os
from quinine import Quinfig, QuinineArgumentParser, tstring, tboolean, tfloat, tinteger, stdict, stlist, default, nullable, required
import torch
from transformers import AutoModel, AutoTokenizer
from pathlib import Path 
from tqdm import tqdm 
import pickle

from preprocess import make_contexts
from data_loader import fetch_wikipedia

N_CONTEXTS = 20
CONTEXT_LENGTH = 1
RESOURCE_NAME = 'wikipedia'

documents = fetch_wikipedia()
make_contexts(documents, CONTEXT_LENGTH, RESOURCE_NAME, N_CONTEXTS)