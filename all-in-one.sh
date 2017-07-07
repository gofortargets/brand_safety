#!/bin/bash -x
CODE_DIR=/home/ta/Projects/brand_safety/svm/preprocess
DATA_DIR=/home/ta/Projects/brand_safety/data

# Step 0: clean data
#rm -rf ${DATA_DIR}/train_preprocess ${DATA_DIR}/test_preprocess
#mkdir -pv ${DATA_DIR}/train_preprocess ${DATA_DIR}/test_preprocess
#python clean_data.py ${DATA_DIR}/train ${DATA_DIR}/train_preprocess
#python clean_data.py ${DATA_DIR}/test ${DATA_DIR}/test_preprocess
#
# Step 1: Prepare text files:
##rm -rf ${DATA_DIR}/train ${DATA_DIR}/test ${DATA_DIR}/train_Vocab-Reduced/ ${DATA_DIR}/test_Vocab-Reduced/
##mkdir -pv ${DATA_DIR}/train ${DATA_DIR}/test ${DATA_DIR}/train_Vocab-Reduced/ ${DATA_DIR}/test_Vocab-Reduced/
##python ${CODE_DIR}/train_test_split.py ${DATA_DIR}/raw ${DATA_DIR}/train ${DATA_DIR}/test 0.1

#rm -rf ${DATA_DIR}/train_Vocab-Reduced/ ${DATA_DIR}/test_Vocab-Reduced/
#mkdir -pv ${DATA_DIR}/train_Vocab-Reduced/ ${DATA_DIR}/test_Vocab-Reduced/

#build vocab
#python ${CODE_DIR}/build_vocab.py ${DATA_DIR}/train_preprocess/ ${DATA_DIR}/vocab.pkl
#for each file, only consider the word in new vocab, discard the others
#python ${CODE_DIR}/reduce_vocab.py ${DATA_DIR}/train_preprocess/ ${DATA_DIR}/train_Vocab-Reduced/ ${DATA_DIR}/vocab.pkl 200000
#python ${CODE_DIR}/reduce_vocab.py ${DATA_DIR}/test_preprocess/ ${DATA_DIR}/test_Vocab-Reduced/ ${DATA_DIR}/vocab.pkl 200000

# Step 2: Preprocess text file and export the pickles

CODE_DIR=/home/ta/Projects/brand_safety/svm
#python ${CODE_DIR}/preprocess.py ${DATA_DIR}/train_Vocab-Reduced \
#                                                                ${DATA_DIR}/test_Vocab-Reduced midf ${DATA_DIR}

# Step 3: Do classification
python ${CODE_DIR}/classify.py ${DATA_DIR}

# Buoc 4: Export the vocab file to deploy on server
#python ${CODE_DIR}/vocab.py ${DATA_DIR}/train.pkl ${DATA_DIR}/full_vocab.pkl
