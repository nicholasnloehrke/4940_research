import ast
import pandas as pd
import numpy as np
import emoji
import re
import spacy
import fasttext


# load models
nlp = spacy.load('en_core_web_sm')
ft_model = fasttext.load_model('crawl-300d-2M-subword.bin')

# load review data
data = pd.read_csv('review_data.csv', nrows=50)

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

# convert tokens to FastText embeddings
def get_embeddings(tokens):
    embeddings = []
    for token in tokens:
        embeddings.append(ft_model.get_word_vector(token))
    return embeddings

data['embeddings'] = data['review'].apply(get_embeddings)

# pad embeddings
max_length = max(data['embeddings'].apply(len))
data['embeddings'] = data['embeddings'].apply(lambda x: x + [[0] * 300] * (max_length - len(x)))

data.to_csv('review_data_with_embeddings.csv')

print(f'Embedding length: {max_length}')