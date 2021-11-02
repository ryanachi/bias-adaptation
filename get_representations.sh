CONDA_ENV="rchi-bert-sa"

CMD="nlprun -a $CONDA_ENV -g 1 -p high -o logs/get_representations/roberta-base-wiki.log -q jag -x jagupard[28-29] -n representations 'python3 representations/get_representations.py'"
eval $CMD
