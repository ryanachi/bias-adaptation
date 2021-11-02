"""
schema.py

This is an example schema.py file for parsing a quinfig config.yaml file.

"""
from typing import Any, Dict

from quinine.common.cerberus import default, merge, nullable, tboolean, tfloat, tinteger, tlist, tstring

def get_eval_schema() -> Dict[str, Any]:
    # Update as Necessary --> see `https://github.com/krandiash/quinine#cerberus-schemas-for-validation`
    schema = {
        "model_name": merge(tstring, nullable, default('bert-base-uncased')),
        "train_dataset": merge(tstring, nullable, default('imdb')),
        "train_datasets_to_evaluate": merge(tlist, nullable),
        "run_id": merge(tstring, nullable, default(None)),
        "bsz": merge(tinteger, nullable, default(24)),
        "seed": merge(tinteger, nullable, default(21)),
        "dataset_size": merge(tstring, nullable, default('full')),
        "patience": merge(tinteger, nullable, default(3)),
        "learning_rate": merge(tfloat, nullable, default(2e-5)),
        "num_train_epochs": merge(tinteger, nullable, default(3)),
        "per_device_train_batch_size": merge(tinteger, nullable, default(16)),
        "num_labels": merge(tinteger, nullable, default(2)),
#         "save_strategy": merge(tstring, nullable, default("epoch")),
        "adaptation_process": merge(tstring, nullable, default("finetuning")),
        "save_strategy": merge(tstring, nullable, default("steps")),
        "evaluation_strategy": merge(tstring, nullable, default("steps")),
        "num_save_checkpoints": merge(tinteger, nullable, default(10)),
        "evals_per_epoch": merge(tinteger, nullable, default(4)),
        "dataset_to_load": merge(tstring, nullable),
        "evaluation_datasets": merge(tlist, nullable),
        "max_checkpoint_step": merge(tinteger, nullable),
        "seeds_to_evaluate": merge(tlist, nullable),
    }

    return schema


