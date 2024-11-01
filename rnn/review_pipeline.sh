#!/bin/bash

python augment_reviews.py data/review_data.csv data/augmented_review_data.csv

python reviews_to_embeddings.py data/augmented_review_data.csv -o data/augmented_embeddings.pkl