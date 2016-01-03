#! /bin/bash

# require:
# python-sklearn
# keras

python trainAcousticModel.py ./fbank/train.ark ./label/train.lab AM
python PredByAM.py ./fbank/train.ark ./label/train.lab ./fbank/test.ark frame.csv AM

python HW1_to_final.py frame.csv ./phone

# here create nbest, all files are already provide
# please name the nbest file as: nbest.txt



python trainLanguageModel.py LM timit_sentence.txt
python EvalProbByLM.py LM nbest.txt sub.csv

