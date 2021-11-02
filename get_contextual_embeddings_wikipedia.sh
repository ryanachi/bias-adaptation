CONDA_ENV="rchi-bert-sa"

CMD="nlprun -a $CONDA_ENV -g 0 -r 32GB -p high -o logs/get_representations/get_contextual_embeddings_wikipedia.log -q john -n contexts 'python3 get_contextual_embeddings_wikipedia.py'"
eval $CMD
