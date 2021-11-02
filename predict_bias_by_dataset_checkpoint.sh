CONDA_ENV="rchi-bert-sa"
SST_BIAS_EVAL_CONF="conf/evaluation/sentiment_analysis/sa-bias-evaluation.yaml"

declare -a MethodsArray=("finetuning" "linear_probing" "bitfit")
declare -a DatasetsArray=("yelp_polarity")


for dataset in ${DatasetsArray[@]}; do
    for method in ${MethodsArray[@]}; do
        CMD="nlprun -a $CONDA_ENV -g 1 -p standard -o logs/predict_bias_by_dataset_checkpoint/$dataset-$method.log -q jag -x jagupard[28-29] -n sst-$dataset 'python3 predict_bias_by_dataset_checkpoint.py --config $SST_BIAS_EVAL_CONF --train_dataset $dataset --adaptation_process $method'"
        eval $CMD
    done
done
