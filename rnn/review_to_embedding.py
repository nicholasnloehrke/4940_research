import pandas as pd
import numpy as np
import emoji
import re
import spacy
import fasttext
import pickle


# load models
nlp = spacy.load('en_core_web_sm')
ft_model = fasttext.load_model('crawl-300d-2M-subword.bin')

# load review data
data = pd.read_csv('data/review_data.csv', nrows=1000)
# data = pd.read_csv('data/review_data.csv')

# remove empty reviews
data = data[data['review'].apply(lambda x: isinstance(x, str))]

# remove numbers from reviews
data['review'] = data['review'].apply(lambda review: re.sub(r'\d+', '', review))

# remove emojis
data['review'] = data['review'].apply(lambda review: emoji.replace_emoji(review, replace=''))

# remove neutral ratings
data = data[data['rating'] != 3]

# create binary sentiment (0 = negative, 1 = positive)
data['sentiment'] = np.where(data['rating'] > 2, 1, 0)

# remove stop words and punctuation
data['review'] = data['review'].apply(
    lambda text: (
        [token.text for token in nlp(text) if not token.is_stop and not token.is_punct]
        if isinstance(text, str)
        else []
    )
)    

# convert tokens to a 2D NumPy array of embeddings
def get_embeddings(tokens):
    embeddings = [ft_model.get_word_vector(token) for token in tokens]
    return np.array(embeddings, dtype=np.float32)

data['embeddings'] = data['review'].apply(get_embeddings)

# pad embeddings to max length
def pad_embeddings(embedding, max_length):
    if len(embedding) < max_length:
        padding = np.zeros((max_length - len(embedding), 300), dtype=np.float32)
        return np.vstack([embedding, padding])
    return embedding

max_length = max(data['embeddings'].apply(len))
data['embeddings'] = data['embeddings'].apply(lambda x: pad_embeddings(x, max_length))

selected_columns = data[['sentiment', 'embeddings']]

selected_columns.to_pickle('data/review_data_embeddings.pkl')
    
print(f'Embedding length: {max_length}')
print(f"Number of sentiments: {len(data['sentiment'])}")
print(f"Number of embeddings: {len(data['embeddings'])}")