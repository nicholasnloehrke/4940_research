import random
import pandas as pd
import numpy as np
import argparse


def random_deletion_with_duplication(review, sentiment, p=0.1):
    """Duplicate the review and sentiment, then apply random deletion."""
    duplicated_reviews = [(review, sentiment)]
    
    words = review.split()
    if len(words) == 0:
        return duplicated_reviews  # Return original if no words
    
    deleted_review = ' '.join([word for word in words if random.uniform(0, 1) > p])
    duplicated_reviews.append((deleted_review, sentiment))
    
    return duplicated_reviews

def add_typo(review, typo_rate=0.1):
    """Introduce random typos into the review."""
    words = list(review)
    n_typos = int(len(words) * typo_rate)
    for _ in range(n_typos):
        idx = random.randint(0, len(words) - 1)
        if words[idx].isalpha():  # Only modify if it's a letter
            char_index = random.randint(0, len(words[idx]) - 1)
            # Replace with a random character
            words[idx] = (words[idx][:char_index] +
                          random.choice('abcdefghijklmnopqrstuvwxyz') +
                          words[idx][char_index + 1:])
    return ''.join(words)

def segment_sentences(review, max_length=20):
    """Segment the review into shorter sentences."""
    words = review.split()
    return [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]

def augment_reviews(input_file, output_file):
    df = pd.read_csv(input_file)
    df = df.dropna()

    new_reviews = []
    new_sentiments = []

    for index, row in df.iterrows():
        
        review = row["review"]
        rating = row["rating"]

        # Append the original review and sentiment
        new_reviews.append(review)
        new_sentiments.append(rating)

        if index % 2 != 0:
            continue

        # Random Deletion with Duplication
        augmented_reviews = random_deletion_with_duplication(review, rating)
        for aug_review, aug_sentiment in augmented_reviews:
            new_reviews.append(aug_review)
            new_sentiments.append(aug_sentiment)

        # Adversarial Example (Typos)
        new_reviews.append(add_typo(review))
        new_sentiments.append(rating)

        # Segmenting Sentences
        segments = segment_sentences(review)
        for segment in segments:
            new_reviews.append(segment)
            new_sentiments.append(rating)

    # Create a new DataFrame with augmented data
    augmented_df = pd.DataFrame({
        "rating": new_sentiments,
        "review": new_reviews
    })
    
    # Save the augmented data to a new CSV file
    augmented_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment Amazon review data.")
    parser.add_argument("input_file", type=str, help="Input CSV file with reviews.")
    parser.add_argument("output_file", type=str, help="Output CSV file for augmented reviews.")
    
    args = parser.parse_args()

    augment_reviews(args.input_file, args.output_file)
