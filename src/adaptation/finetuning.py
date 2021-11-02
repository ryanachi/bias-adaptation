"""
finetune.py

finetuning script called by train.py
"""

from datasets import load_metric
from datetime import datetime
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from .my_trainer import EarlyStopTrainer as Trainer

STORAGE_FOLDER = "/u/scr/nlp/mercury/bias-adaptation"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def finetune(quinfig, dataset_splits):
    train_dataset, test_dataset = dataset_splits
    
    # Create Unique Run Name
    run_id = quinfig.run_id
    if run_id is None:
        run_id = f"placeholder+{datetime.now().strftime('%Y-%m-%d-%H:%M')}"
    
    # Finetuning
    training_args = TrainingArguments("test_trainer")
    training_args.output_dir = f"{STORAGE_FOLDER}/sa/{quinfig.model_name}/{quinfig.train_dataset}/{run_id}"
    training_args.seed = quinfig.seed
    training_args.learning_rate = quinfig.learning_rate
    training_args.num_train_epochs = quinfig.num_train_epochs
    training_args.per_device_train_batch_size = quinfig.per_device_train_batch_size
    training_args.save_strategy = quinfig.save_strategy
    training_args.evaluation_strategy = quinfig.evaluation_strategy
    training_args.save_steps = 0.30 * quinfig.train_set_size / quinfig.per_device_train_batch_size 
    training_args.eval_steps = 0.25 * quinfig.train_set_size / quinfig.per_device_train_batch_size
    training_args.patience = 0
    
    print(f"\nTrain set size: {quinfig.train_set_size}") # specified when loading data
    print(f"Train set steps: {quinfig.train_set_size / quinfig.per_device_train_batch_size}")
    print(f"Saving every {training_args.save_steps} steps")
    print(f"Evaluating every {training_args.eval_steps} steps\n")
    
    metric = load_metric("accuracy", cache_dir=f"{STORAGE_FOLDER}/metrics")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def model_init():
        model = AutoModelForSequenceClassification.from_pretrained(quinfig.model_name, num_labels=quinfig.num_labels, cache_dir=f"{STORAGE_FOLDER}/hf_transformers")
        model.to(device)
        
        if quinfig.adaptation_process == "finetuning":
            pass
        
        elif quinfig.adaptation_process == "linear_probing":
            if quinfig.model_name.startswith("roberta"):
                for param in model.roberta.parameters():
                    param.requires_grad = False

            elif quinfig.model_name.startswith("bert"):
                for param in model.bert.parameters():
                    param.requires_grad = False
                    
        elif quinfig.adaptation_process == "bitfit":
            if quinfig.model_name.startswith("roberta"):
                for n, p in model.roberta.named_parameters():
                    if "bias" in n:
                        p.requires_grad = False

            elif quinfig.model_name.startswith("bert"):
                for n, p in model.bert.named_parameters():
                    if "bias" in n:
                        p.requires_grad = False
           
        return model
    
    
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    trainer.save_model(f"{STORAGE_FOLDER}/sa/{quinfig.model_name}/{quinfig.train_dataset}/{run_id}")
    
    trainer.evaluate(eval_dataset=test_dataset)