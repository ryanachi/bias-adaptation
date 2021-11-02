from datasets import load_metric
import numpy as np
import os
from quinine import Quinfig, QuinineArgumentParser, tstring, tboolean, tfloat, tinteger, stdict, stlist, default, nullable, required
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
# from .my_trainer import EarlyStopTrainer as Trainer

STORAGE_FOLDER = "/u/scr/nlp/mercury/bias-adaptation"
CHECKPOINT_FOLDER = f"{STORAGE_FOLDER}/sa/roberta-base"
EEC_DATASET_FOLDER = f"{STORAGE_FOLDER}/datasets/EEC"
LOG_FOLDER = f"/sailhome/ryanchi/bias/logs"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def analyze_bias(quinfig, dataset_splits):
    male_eec_dataset, female_eec_dataset = dataset_splits
    
    model_loaded = AutoModelForSequenceClassification.from_pretrained(f"{CHECKPOINT_FOLDER}/{quinfig.train_dataset}/{quinfig.run_id}/checkpoint-{quinfig.max_checkpoint_step}")
    model_loaded.to(device)
    
    trainer = Trainer(model=model_loaded)
    
    male_eec_predictions = trainer.predict(male_eec_dataset)
    female_eec_predictions = trainer.predict(female_eec_dataset)
    
    path = os.path.join(f"{CHECKPOINT_FOLDER}/{quinfig.train_dataset}/{quinfig.run_id}/checkpoint-{quinfig.max_checkpoint_step}", "predictions")
    os.makedirs(path, exist_ok=True)
    
    # "max_checkpoint_step" should be renamed, as it's not necessarily the max checkpoint each instance it's used
    torch.save(male_eec_predictions, f"{CHECKPOINT_FOLDER}/{quinfig.train_dataset}/{quinfig.run_id}/checkpoint-{quinfig.max_checkpoint_step}/predictions/male_eec_predictions.pt")
    torch.save(female_eec_predictions, f"{CHECKPOINT_FOLDER}/{quinfig.train_dataset}/{quinfig.run_id}/checkpoint-{quinfig.max_checkpoint_step}/predictions/female_eec_predictions.pt")
    
    
def get_dataset_performance(quinfig, dataset_splits):
    train_dataset, test_dataset = dataset_splits

    model_loaded = AutoModelForSequenceClassification.from_pretrained(f"{CHECKPOINT_FOLDER}/{quinfig.train_dataset}/{quinfig.run_id}/checkpoint-{quinfig.max_checkpoint_step}")
    model_loaded.to(device)
    
    metric = load_metric("accuracy", cache_dir=f"{STORAGE_FOLDER}/metrics")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    trainer = Trainer(model=model_loaded, compute_metrics=compute_metrics)
    test_predictions = trainer.predict(test_dataset)
    
    path = os.path.join(f"{CHECKPOINT_FOLDER}/{quinfig.train_dataset}/{quinfig.run_id}", "predictions")
    os.makedirs(path, exist_ok=True)
    torch.save(test_predictions, f"{CHECKPOINT_FOLDER}/{quinfig.train_dataset}/{quinfig.run_id}/predictions/{quinfig.dataset_to_load}_predictions.pt")
        
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print(f'Metrics for {quinfig.dataset_to_load}: {test_metrics}\n\n')
    
    #Temporary fix for 8/3/21
    with open(f"{LOG_FOLDER}/{quinfig.train_dataset}/{quinfig.run_id}-{quinfig.max_checkpoint_step}-accuracy", 'a') as g:
#     with open(f"{LOG_FOLDER}/{quinfig.train_dataset}/{quinfig.run_id}-accuracy", 'a') as g:
        g.write(f'Metrics for {quinfig.dataset_to_load}: {test_metrics}\n\n')

    g.close
    
    return test_metrics