#! /bin/bash

# 2015 spring MLDS HW3 
# Kunan
#  b00901146 Chu Po Hsien
#  b00901010 Lin Xien Jing
#  Kado

# sudo pip install gensim
git clone https://github.com/stegben/MLDS_HW3.git

cd ./MLDS_HW3
# training data preprocessing
python training_preprocessing.py Holmes_Training_Data tr.txt

# create word2vec
cd ./word2vec
make
time ./word2vec -train ../tr.txt -output vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15

# train model and predict
cd ../
python HW3.py Holmes_Training_Data testing_data.txt ./word2vec/vectors.bin output.csv
