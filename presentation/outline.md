# UT Dallas Presentation Outline

### Goal

Sentiment analysis on Amazon review data

### Model pipeline

- Reviews
    - Augmentation
        - Shuffling
        - Chunking
- Topology overview
    - Use Fasttext as feature extractor ***(Am I using 'feature extractor' correctly here?)***
    - Use RNN as feature extractor
    - FFNN as final classifier
- Embeddings - Fasttext  ***(PICTURE)***
    - Shortcomings of Word2vec
    - Trained using CBOW on common crawl ***(PICTURE)***
    - $n$-gram's for OOV words
        - subword of 'refrigerator': (['refrigerator', '<ref', 'refr', 'efri', 'frig', 'rige', 'iger', 'gera', 'erat', 'rato', 'ator', 'tor>'], array([6315, 3998276, 
3822544, 3278539, 2069117, 3246884, 3006258,
       3159920, 2716211, 3195125, 3616757, 3672916]))
- RNN
    - Hyperparameter's
    - Topology
        - LSTM (maybe mention )
        - Fully connect layers based on last hidden state
        - Output layer (2 classes)
    - Training
        - 
- FFNN
    - Hyperparameter's
    - Topology
        - Input layer ordering (hidden states), padding (0-vectors)
        - Relu
        - Output layer (2 classes)
- Technology
    - Python
    - PyTorch
        - Almost entirely because GPU was easier to setup
    - Bash