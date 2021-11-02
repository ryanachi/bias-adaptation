#!/bin/bash

CONDA_ENV="rchi-bert-sa"
IMDB_FINETUNING_CONF="~/bias/conf/finetuning/sentiment_analysis/imdb-finetuning.yaml"
IMDB_BITFIT_CONF="~/bias/conf/finetuning/sentiment_analysis/imdb-bitfit.yaml"
IMDB_PROBING_CONF="~/bias/conf/finetuning/sentiment_analysis/imdb-probing.yaml"
IMDB_ADAPTATION_CONF="~/bias/conf/finetuning/sentiment_analysis/imdb-adaptation.yaml"
SST_ADAPTATION_CONF="~/bias/conf/finetuning/sentiment_analysis/sst-adaptation.yaml"
YELP_ADAPTATION_CONF="~/bias/conf/finetuning/sentiment_analysis/yelp-adaptation.yaml"
AMAZON_ADAPTATION_CONF="~/bias/conf/finetuning/sentiment_analysis/amazon-adaptation.yaml"

declare -a StringArray=("finetuning" "bitfit" "linear_probing")

# for i in {4..5}
for i in {1..5}
    do
       for val in ${StringArray[@]}; do
#            CMD="nlprun -a $CONDA_ENV -g 1 -p standard -o logs/sst/$val-$i.log -q jag -n SST-$val-$i -x jagupard[28-29] 'python3 train.py --config $SST_ADAPTATION_CONF --seed $i --run_id $val-seed-$i --adaptation_process $val'"
#            CMD="nlprun -a $CONDA_ENV -g 1 -p standard -o logs/imdb/$val-$i.log -q jag -n IMDB-$val-$i -x jagupard[28-29] 'python3 train.py --config $IMDB_ADAPTATION_CONF --seed $i --run_id $val-seed-$i --adaptation_process $val'"
           CMD="nlprun -a $CONDA_ENV -g 1 -p standard -o logs/yelp/$val-$i.log -q jag -n YELP$i-$val -x jagupard[28-29] 'python3 train.py --config $YELP_ADAPTATION_CONF --seed $i --run_id $val-seed-$i --adaptation_process $val'"
            eval $CMD
        done
    done
