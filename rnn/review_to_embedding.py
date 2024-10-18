import fasttext
import numpy as np
import pandas as pd
import re
import spacy
import argparse


nlp = None
ft = None

def load_models():
    """Used to lazy-load models after parsing arguments"""
    global nlp, ft
    nlp = spacy.load("en_core_web_sm")
    ft = fasttext.load_model("models/crawl-300d-2M-subword.bin")


def clean_text(text):
    """Remove urls, html tags, special characters, and numbers"""
    text = re.sub(r"http\S+|www\S+|<.*?>", "", text)  # remove urls and html tags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove special characters and numbers
    text = text.strip().lower()
    return text


def tokenize(text):
    """Tokenize with lemmatization and removing stop words"""
    cleaned_text = clean_text(text)
    doc = nlp(cleaned_text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)


def get_embeddings(tokens):
    """Convert tokens to a numpy array of embeddings"""
    embeddings = [ft.get_word_vector(token) for token in tokens]
    return np.array(embeddings, dtype=np.float32)


def pad_embedding(embeddings, pad_length):
    """Pad embeddings to the specified maximum length."""
    if len(embeddings) == 0:
        return np.zeros((pad_length, 300), dtype=np.float32)
    elif len(embeddings) < pad_length:
        padding = np.zeros((pad_length - len(embeddings), 300), dtype=np.float32)
        return np.vstack([embeddings, padding])
    return embeddings[:pad_length]


def main(args):
    df = pd.read_csv(args.input_file, nrows=args.rows) if args.rows > 0 else pd.read_csv("data/review_data.csv")
    df = df.dropna()

    df["sentiment"] = np.where(df["rating"] < 3, 0, 1)
    df["cleaned_review"] = df["review"].apply(tokenize)
    df["embedding"] = df["cleaned_review"].apply(get_embeddings)
    
    if args.pad_length > 0:
        df["embedding"] = df["embedding"].apply(lambda x: pad_embedding(x, args.pad_length))
    
    output_df = df[["sentiment", "embedding"]]
    
    output_df.to_pickle(args.output)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process review data.")
    parser.add_argument("-r", "--rows", type=int, default=0, help="Number of rows to read from the CSV file.")
    parser.add_argument("-p", "--pad_length", type=int, default=0, help="Maximum length for padding embeddings.")
    parser.add_argument("-o", "--output", type=str, default="embedding.pkl", help="Output file")
    parser.add_argument("input_file", type=str, help="Input CSV file")
    
    args = parser.parse_args()
    
    load_models()

    main(args)
