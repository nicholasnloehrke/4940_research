#!/bin/bash

#####################################################################################
#                                    TRAINING                                       #
#####################################################################################

# Create training/test splits
python -m scripts.create_train_test_datasets data/raw_reviews.csv --train_outfile data/raw_training_reviews.csv --test_outfile data/raw_test_reviews.csv --test_size 0.15

# Preprocess
python -m scripts.preprocess_reviews data/raw_training_reviews.csv -o data/preprocessed_training_reviews.csv --chunk 20 --max_chunks 5 --shuffle_chunk 20

# Generate training embeddings
python -m scripts.generate_embeddings fasttext/crawl-300d-2M-subword.bin data/preprocessed_training_reviews.csv -o data/training_embeddings.pkl

# Train RNN
python -m scripts.train_rnn data/training_embeddings.pkl -o data/test_rnn.model

# Extract hidden states
python -m scripts.extract_hidden_states data/test_rnn.model data/training_embeddings.pkl -d 200 -o data/training_hidden_states.pkl

# Train FFNN
python -m scripts.train_ffnn data/training_hidden_states.pkl -o data/test_ffnn.model

#####################################################################################
#                                   EVALUATION                                      #
#####################################################################################

# Generate test embeddings
python -m scripts.generate_embeddings fasttext/crawl-300d-2M-subword.bin data/raw_test_reviews.csv -o data/test_embeddings.pkl

# Evaluate RNN
python -m scripts.evaluate_rnn data/test_rnn.model data/test_embeddings.pkl

# Extract hidden states
python -m scripts.extract_hidden_states data/test_rnn.model data/test_embeddings.pkl -d 200 -o data/raw_test_hidden_states.pkl

# Evaluate FFNN
python -m scripts.evaluate_ffnn data/test_ffnn.model data/raw_test_hidden_states.pkl