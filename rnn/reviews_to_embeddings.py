import numpy as np
import pandas as pd
import argparse
import fasttext
from pathlib import Path


FASTTEXT_MODEL_PATH = str(Path("pretrained_models/crawl-300d-2M-subword.bin").resolve())

ft = None

def load_models():
    """Used to lazy-load models after parsing arguments"""
    global ft
    ft = fasttext.load_model(FASTTEXT_MODEL_PATH)

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
    df = pd.read_csv(args.input_file, nrows=args.rows) if args.rows > 0 else pd.read_csv(args.input_file)
    df = df.dropna()
    df["sentiment"] = np.where(df["rating"] < 3, 0, 1)
    df["embedding"] = df["review"].apply(get_embeddings)    

    if args.pad_length > 0:
        df["embedding"] = df["embedding"].apply(lambda x: pad_embedding(x, args.pad_length))
    
    output_df = df[["sentiment", "embedding"]]
    
    output_df.to_pickle(args.output)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process review data.")
    parser.add_argument("-r", "--rows", type=int, default=0, help="Number of rows to read from the CSV file.")
    parser.add_argument("-p", "--pad_length", type=int, default=0, help="Maximum length for padding embeddings.")
    parser.add_argument("-o", "--output", type=str, default="embeddings.pkl", help="Output file")
    parser.add_argument("input_file", type=str, help="Input CSV file")
    
    args = parser.parse_args()

    load_models()

    main(args)
