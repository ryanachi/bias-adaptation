from collections import OrderedDict
from scipy import stats
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

CORRELATIONS_FOLDER = "/sailhome/ryanchi/bias/check_correlations"
SAVE_FOLDER = f"{CORRELATIONS_FOLDER}/img_errbars_no_probing_w_id"

bias_tradeoffs_sst2 = pd.read_csv(f"{CORRELATIONS_FOLDER}/bias adaptation tradeoffs - Correlations_ SST2.csv")
bias_tradeoffs_imdb = pd.read_csv(f"{CORRELATIONS_FOLDER}/bias adaptation tradeoffs - Correlations_ IMDB.csv")
bias_tradeoffs_yelp = pd.read_csv(f"{CORRELATIONS_FOLDER}/bias adaptation tradeoffs - Correlations_ Yelp.csv")

#drop rows with empty cells
bias_tradeoffs_sst2 = bias_tradeoffs_sst2.dropna()
bias_tradeoffs_imdb = bias_tradeoffs_imdb.dropna()
bias_tradeoffs_yelp = bias_tradeoffs_yelp.dropna()

#insert name of train set
bias_tradeoffs_sst2.insert(0, 'train_dataset', 'sst2')
bias_tradeoffs_imdb.insert(0, 'train_dataset', 'imdb')
bias_tradeoffs_yelp.insert(0, 'train_dataset', 'yelp_polarity')

df = pd.concat([bias_tradeoffs_sst2, bias_tradeoffs_imdb, bias_tradeoffs_yelp])

#normalization
def get_min_max_score(curr_val, min_val, max_val):
    if (max_val - min_val) != 0:
        return (curr_val - min_val) / (max_val - min_val)
    return (curr_val - min_val)

TRAIN_DATASETS = ["sst2", "imdb", "yelp_polarity"]
ADAPTATION_METHODS = ["finetuning", "bitfit"]
METRICS = ["DP", "EO (y=0)", "EO (y=1)", "OOD", "ID"]
#each of the first three metrics is understood to be M-F
#fourth is average accuracy across train dataset + two OOD datasets

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
LINESTYLE_TUPLES = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
LINESTYLES = ["-", "--", "-.", ":", LINESTYLE_TUPLES["densely dashdotted"]]
ADAPTATION_COLORS = ["b", "g"]
METRIC_COLORS = ["c", "y", "m", "b", "r"]
MIN_SEED = 1
MAX_SEED = 5

os.makedirs(f"{SAVE_FOLDER}/by_dataset_and_metric/single_seed", exist_ok=True)
os.makedirs(f"{SAVE_FOLDER}/by_dataset_and_metric/across_seeds", exist_ok=True)

os.makedirs(f"{SAVE_FOLDER}/by_dataset_and_method/single_seed", exist_ok=True)
os.makedirs(f"{SAVE_FOLDER}/by_dataset_and_method/across_seeds", exist_ok=True)

for train_dataset in TRAIN_DATASETS:
    print(f"Train dataset: {train_dataset}")

    df_curr_dataset = df[df["train_dataset"]==train_dataset] #copy?
    
    # by_dataset_and_metric
    for metric in METRICS:     
        # graph seeds separately
        plt.clf()
        for seed, linestyle in zip(range(MIN_SEED, MAX_SEED + 1), LINESTYLES):
            df_curr_seed = df_curr_dataset[df_curr_dataset["seed"]==seed].copy()
            for adaptation_method, adaptation_color in zip(ADAPTATION_METHODS, ADAPTATION_COLORS):
                df_curr = df_curr_seed[df_curr_seed["adaptation"]==adaptation_method].copy()
                plt.plot(df_curr['checkpoint_step'], df_curr[metric], linestyle=linestyle, color=adaptation_color, label=f"{adaptation_method} {seed}")
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.savefig(f'{SAVE_FOLDER}/by_dataset_and_metric/single_seed/{train_dataset}_{metric}.png', bbox_inches='tight')
        
        # graph inter-seed averages
        plt.clf()
        across_seeds_df = df_curr_dataset.groupby(['train_dataset', 'adaptation', 'checkpoint_step']).mean()
        across_seeds_df = across_seeds_df.reset_index()
        
        across_seeds_std_df = df_curr_dataset.groupby(['train_dataset', 'adaptation', 'checkpoint_step']).std()
        across_seeds_std_df = across_seeds_std_df.reset_index()
        
        across_seeds_df[f"{metric} STD"] = across_seeds_std_df[f"{metric}"]
        
        for adaptation_method, adaptation_color in zip(ADAPTATION_METHODS, ADAPTATION_COLORS):
            df_curr = across_seeds_df[across_seeds_df["adaptation"]==adaptation_method].copy()
            plt.errorbar(df_curr['checkpoint_step'], df_curr[metric], yerr=df_curr[f"{metric} STD"], color=adaptation_color, label=f"{adaptation_method} {seed}")
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.savefig(f'{SAVE_FOLDER}/by_dataset_and_metric/across_seeds/{train_dataset}_{metric}.png', bbox_inches='tight')
        
    # by_dataset_and_method
    for adaptation_method in ADAPTATION_METHODS:
        print(f"Adaptation method: {adaptation_method}")
        df_curr_method = df_curr_dataset[df_curr_dataset['adaptation']==adaptation_method].copy() 
        df_curr_method['DP norm'] = df_curr_method.apply(lambda row: get_min_max_score(
                                    row['DP'],
                                    min(df_curr_method['DP']),
                                    max(df_curr_method['DP'])), axis=1)
        df_curr_method['EO (y=0) norm'] = df_curr_method.apply(lambda row: get_min_max_score(
                                    row['EO (y=0)'],
                                    min(df_curr_method['EO (y=0)']),
                                    max(df_curr_method['EO (y=0)'])), axis=1)
        df_curr_method['EO (y=1) norm'] = df_curr_method.apply(lambda row: get_min_max_score(
                                    row['EO (y=1)'],
                                    min(df_curr_method['EO (y=1)']),
                                    max(df_curr_method['EO (y=1)'])), axis=1)
        df_curr_method['OOD norm'] = df_curr_method.apply(lambda row: get_min_max_score(
                                    row['OOD'],
                                    min(df_curr_method['OOD']),
                                    max(df_curr_method['OOD'])), axis=1)
        df_curr_method['ID norm'] = df_curr_method.apply(lambda row: get_min_max_score(
                                    row['ID'],
                                    min(df_curr_method['ID']),
                                    max(df_curr_method['ID'])), axis=1)
        # graph seeds separately
        plt.clf()
        for seed, linestyle in zip(range(MIN_SEED, MAX_SEED + 1), LINESTYLES):
            df_curr = df_curr_method[df_curr_method["seed"]==seed]
            for metric, metric_color in zip(METRICS, METRIC_COLORS):
                plt.plot(df_curr['checkpoint_step'], df_curr[f"{metric} norm"], linestyle=linestyle, color=metric_color, label=f"{metric} {seed}")
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.savefig(f'{SAVE_FOLDER}/by_dataset_and_method/single_seed/{train_dataset}_{adaptation_method}.png', bbox_inches='tight') 
        
        # graph inter-seed averages
        plt.clf()
        across_seeds_df = df_curr_method.groupby(['train_dataset', 'adaptation', 'checkpoint_step']).mean()
        across_seeds_df = across_seeds_df.reset_index()
        
        across_seeds_std_df = df_curr_method.groupby(['train_dataset', 'adaptation', 'checkpoint_step']).std()
        across_seeds_std_df = across_seeds_std_df.reset_index()
        
        for metric, metric_color in zip(METRICS, METRIC_COLORS):
            across_seeds_df[f"{metric} STD"] = across_seeds_std_df[f"{metric}"]
            plt.errorbar(across_seeds_df['checkpoint_step'], across_seeds_df[f"{metric} norm"], yerr=across_seeds_df[f"{metric} STD"], color=metric_color, label=f"{metric}")
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.savefig(f'{SAVE_FOLDER}/by_dataset_and_method/across_seeds/{train_dataset}_{adaptation_method}.png', bbox_inches='tight')
        
        
        