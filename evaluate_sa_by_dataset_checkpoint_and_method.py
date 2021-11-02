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
from src.analysis.evaluate_dataset import get_dataset_performance

LOG_FOLDER = "/sailhome/ryanchi/bias/logs"
STORAGE_FOLDER = "/u/scr/nlp/mercury/bias-adaptation"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_sa() -> None:
    # Parse Quinfig (via Quinine Argparse Binding)
    quinfig = QuinineArgumentParser(schema=get_eval_schema()).parse_quinfig()
    
    all_seeds_checkpoints_results_list = [] # to be converted into a Pandas DF
    
    if quinfig.train_dataset == "yelp_polarity":
        save_steps = 21_000
    elif quinfig.train_dataset == "sst2":
        save_steps = 2_520
    elif quinfig.train_dataset == "imdb":
        save_steps = 2_500
        
    df_columns = ['adaptation', 'seed', 'checkpoint_step']
    for dataset in quinfig.evaluation_datasets: df_columns.append(dataset)
            
    for run_id in [2, 4, 5]:
        quinfig.run_id = f"{quinfig.adaptation_process}-seed-{run_id}"
        results_for_curr_run_id = []
        
        for checkpoint_number in range(1, 11): # should be 11; stop if it doesn't work
            checkpoint = checkpoint_number * save_steps
            quinfig.max_checkpoint_step = checkpoint # should be renamed to "curr_checkpoint," not "max_checkpoint"
            
            results_for_curr_checkpoint = [quinfig.adaptation_process, run_id, checkpoint]
            
            for dataset in quinfig.evaluation_datasets: #sst2, imdb, yelp_polarity
                    
                quinfig.dataset_to_load = dataset
                dataset_splits = load_dataset_splits(quinfig=quinfig, dataset_cache_dir=f"{STORAGE_FOLDER}/datasets/huggingface", tokenizer_cache_dir=f"{STORAGE_FOLDER}/tokenizer") # (train, dev, test)
                test_metrics = get_dataset_performance(quinfig, dataset_splits)
                results_for_curr_checkpoint.append(test_metrics['eval_accuracy'])
            
            results_for_curr_run_id.append(results_for_curr_checkpoint)
            all_seeds_checkpoints_results_list.append(results_for_curr_checkpoint)
        
        results_for_curr_run_id_df = pd.DataFrame(results_for_curr_run_id, columns=df_columns)
        results_for_curr_run_id_df.to_csv(f'{LOG_FOLDER}/csv/{quinfig.train_dataset}/{quinfig.run_id}-sa.csv')
                    
#     results_df = pd.DataFrame(all_seeds_checkpoints_results_list, columns=df_columns)
#     results_df.to_csv(f'{LOG_FOLDER}/csv/{quinfig.train_dataset}-{quinfig.adaptation_process}-checkpoints-sa.csv')
                
if __name__ == "__main__":
    evaluate_sa()