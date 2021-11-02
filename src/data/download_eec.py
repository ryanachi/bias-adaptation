from datasets import load_dataset, load_metric
import numpy as np
import pandas as pd
from quinine import Quinfig
import torch
from transformers import AutoTokenizer
import random

STORAGE_FOLDER = "/u/scr/nlp/mercury/bias-adaptation"
EEC_FOLDER = f"{STORAGE_FOLDER}/datasets/EEC"
tokenizer = AutoTokenizer.from_pretrained("roberta-base", cache_dir=f"{STORAGE_FOLDER}/tokenizer")

class EECDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 🛠 Prepare the EEC data, no need to dig into this line
eec_df = pd.read_csv(f'{EEC_FOLDER}/Equity-Evaluation-Corpus/Equity-Evaluation-Corpus.csv')

# Remove the sentences for evaluating racial bias
gender_eec_df = eec_df[eec_df['Race'].isna()][:]

# Create identifier to mach sentence pairs
# The EEC data comes withot this matching
MALE_PERSONS = ('he', 'this man', 'this boy', 'my brother', 'my son', 'my husband',
                'my boyfriend', 'my father', 'my uncle', 'my dad', 'him')

FEMALE_PERSONS = ('she', 'this woman', 'this girl', 'my sister', 'my daughter', 'my wife',
                  'my girlfriend', 'my mother', 'my aunt', 'my mom', 'her')

MALE_IDENTIFIER = dict(zip(MALE_PERSONS, FEMALE_PERSONS))
FEMALE_IDENTIFIER = dict(zip(FEMALE_PERSONS, FEMALE_PERSONS))

PERSON_MATCH_WORDS = {**MALE_IDENTIFIER,
                      **FEMALE_IDENTIFIER}

gender_eec_df['PersonIdentifier'] = gender_eec_df['Person'].map(PERSON_MATCH_WORDS)

gender_eec_df = gender_eec_df.sort_values(['Gender', 'Template', 'Emotion word', 'PersonIdentifier'])

gender_split_index = len(gender_eec_df) // 2

# Create two DataFrames, one for 
female_eec_df = gender_eec_df[:gender_split_index].reset_index(False)
male_eec_df = gender_eec_df[gender_split_index:].reset_index(False)

male_eec_texts = male_eec_df.loc[:, 'Sentence'].tolist()
female_eec_texts = female_eec_df.loc[:, 'Sentence'].tolist()

male_eec_encodings = tokenizer(male_eec_texts, truncation=True, padding=True)
female_eec_encodings = tokenizer(female_eec_texts, truncation=True, padding=True)

# so that casting to EECDataset works out ... 
dummy_labels = [0 for i in range(len(male_eec_texts))]

male_eec_dataset = EECDataset(male_eec_encodings, dummy_labels)
female_eec_dataset = EECDataset(female_eec_encodings, dummy_labels)

torch.save(male_eec_dataset, f"{EEC_FOLDER}/male_eec_dataset.pt")
torch.save(female_eec_dataset, f"{EEC_FOLDER}/female_eec_dataset.pt")