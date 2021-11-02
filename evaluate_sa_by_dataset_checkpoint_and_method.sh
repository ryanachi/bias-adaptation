CONDA_ENV="rchi-bert-sa"
SA_EVAL_CONF="conf/evaluation/sentiment_analysis/evaluate_sa_by_dataset_and_method.yaml"

# declare -a MethodsArray=("finetuning" "linear_probing" "bitfit")
# declare -a DatasetsArray=("sst2" "imdb" "yelp_polarity")
declare -a MethodsArray=("finetuning")
declare -a DatasetsArray=("sst2")

for dataset in ${DatasetsArray[@]}; do
    for method in ${MethodsArray[@]}; do
        CMD="nlprun -a $CONDA_ENV -g 1 -p standard -o logs/evaluate_sa_by_dataset_checkpoint_and_method/$dataset-$method-checkpoints.log -q jag -x jagupard[28-29] -n $dataset-$method 'python3 evaluate_sa_by_dataset_checkpoint_and_method.py --config $SA_EVAL_CONF --train_dataset $dataset --adaptation_process $method'"
        eval $CMD
    done
done
