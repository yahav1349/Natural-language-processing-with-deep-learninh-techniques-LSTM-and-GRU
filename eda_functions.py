from collections import Counter
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer

stopwords = set([
    "the","a","and","is","to","in","of","i","you","this","that","it","for","on","with","my","me","feel", "feeling",
    "im", "really", "know", "time", "little", "thing", "now", "way", "people", "make", "one", "will", "ive", "day"
])

def clean_text_basic(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stopwords]
    return words

def most_common_words_per_class(df, classes, top_n=20):
    """
    df: dataframe with text + label
    classes: dict mapping int -> class name
    """
    results = {}

    for label_int, label_name in classes.items():
        subset = df[df['label'] == label_int]['text'].astype(str)

        words = []
        for t in subset:
            t = clean_text_basic(t)
            # words.extend([w for w in t.split() if w not in stopwords and len(w) > 2])
            words.extend(t)

        counter = Counter(words).most_common(top_n)
        results[label_name] = counter

    return results

def linguistic_stats(df, classes):
    df['n_words'] = df['text'].apply(lambda x: len(x.split()))
    df['n_chars'] = df['text'].apply(len)
    df['label_name'] = df['label'].map(classes)
    return df.groupby('label_name')[['n_words','n_chars']].mean()

def unigrarm(df, count):
    tokens = df['text'].apply(clean_text_basic)
    all_words = [word for sublist in tokens for word in sublist]

    words_count = Counter(all_words).most_common(count)
    print('The 5 most common words are:')
    for word, count in words_count:
        print(f"{word}: {count}")

def bigrams(df, top_n=5):
    vec = CountVectorizer(ngram_range=(2,2))
    X = vec.fit_transform(df['text'])
    bigram_counts = X.sum(axis=0)  # sum all bigrams
    bigram_counts = [(word, bigram_counts[0, idx]) for word, idx in vec.vocabulary_.items()]
    bigram_counts = sorted(bigram_counts, key=lambda x: x[1], reverse=True)
    print('The 5 most common bigrams are:')
    for word, count in bigram_counts[:top_n]:
        print(f"{word}: {count}")