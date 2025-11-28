# Twitter Emotion Detection with LSTM and GloVe

## Overview
This project implements a simple LSTM and GRU text classifiers for detecting emotions in Twitter messages. It leverages pre-trained GloVe embeddings to represent words as dense vectors.

## Features
- Preprocessing of Twitter text (tokenization)
- Embedding layer initialized with GloVe vectors
- LSTM and GRU based classifiers for multi-class emotion prediction
- Training loop with loss and accuracy monitoring
- Support for saving and loading model weights

## Setup

1. **Download GloVe embeddings** (e.g., 200d):
```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
*You can also download the GloVe embeddings manually from [Stanford NLP GloVe page](https://nlp.stanford.edu/projects/glove.) zip name: glove.6B.zip*
