#!/bin/bash

# python -m scripts.generate_embeddings data/raw_reviews.csv -o data/raw_review_embeddings.pkl

# python -m scripts.train_rnn data/raw_review_embeddings.pkl -o data/rnn_model.model

# python -m scripts.extract_hidden_states data/rnn_model.model data/raw_review_embeddings.pkl -d 200 -o data/hidden_states.pkl

# python -m scripts.train_ffnn data/hidden_states.pkl -o data/ffnn_model.model

python -m scripts.augment_reviews data/raw_reviews.csv data/augmented_reviews.csv

python -m scripts.generate_embeddings data/augmented_reviews.csv -o data/augmented_review_embeddings.pkl

python -m scripts.train_rnn data/augmented_review_embeddings.pkl -o data/rnn_model.model

python -m scripts.extract_hidden_states data/rnn_model.model data/augmented_review_embeddings.pkl -d 200 -o data/hidden_states.pkl

python -m scripts.train_ffnn data/hidden_states.pkl -o data/ffnn_model.model