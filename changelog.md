### 11/15/2024
- added script to split reviews into test and train sets
- renamed augment_reviews.py -> preprocess_reviews.py
    - the preprocess script now optionally augments reviews and creates a boolean "augmented" column
- training the FFNN and RNN now properly handle the augmented reviews (train on OR and AR, evaluate on OR)


### TODO
- Create different hidden state extraction techniques (truncated, butted, windowed)
- Batch RNN training
- Switch to a configuration file instead of importing config.py
- Bayesian optimization
- Add extra hidden layer dimension for padding flag