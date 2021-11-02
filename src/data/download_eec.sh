EEC_FOLDER = "/u/scr/nlp/mercury/bias-adaptation/datasets/EEC"

wget https://learn.responsibly.ai/word-embedding/data/Equity-Evaluation-Corpus.zip \
     -O $EEC_FOLDER/Equity-Evaluation-Corpus.zip -q

wget https://learn.responsibly.ai/word-embedding/data/SemEval2018-Task1-all-data.zip \
     -O $EEC_FOLDER/SemEval2018-Task1-all-data.zip -q

unzip -qq -o $EEC_FOLDER/Equity-Evaluation-Corpus.zip -d $EEC_FOLDER

unzip -qq -o $EEC_FOLDER/SemEval2018-Task1-all-data.zip -d $EEC_FOLDER

python3 download_eec.py