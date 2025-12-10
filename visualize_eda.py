import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from eda_functions import clean_text_basic
import pandas as pd
from matplotlib.ticker import PercentFormatter


def plot_class_distribution_compare(train_df, test_df, classes):
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df['set name'] = 'Train'
    test_df['set name'] = 'Test'
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # change labels to strings
    df['label_name'] = df['label'].map(classes)

    counts = df.groupby(['set name', 'label_name'])['label'].count().reset_index(name='count')
    
    total_per_set = counts.groupby('set name')['count'].transform('sum')
    
    # Calculate proportion
    counts['proportion'] = counts['count'] / total_per_set

    plt.figure(figsize=(10,6))
    
    # descending order
    order = df['label_name'].value_counts().index

    sns.barplot(
        data=counts,
        x='label_name',
        y='proportion',
        hue='set name',         
        palette=['#4C72B0', '#DA70D6'],
        order=order
    )

    plt.title(f"Train vs Test Class Distribution (Normalized)")
    plt.xlabel("Class")
    plt.ylabel("Proportion")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend(title=None)
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1)) 
    plt.show()
    

def tweet_length_distribution_side_by_side(train_df, test_df):

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['tweet_length'] = train_df['text'].apply(lambda x: len(x.split()))
    test_df['tweet_length'] = test_df['text'].apply(lambda x: len(x.split()))

    stats = {
        'Train': (train_df['tweet_length'].mean(), train_df['tweet_length'].std()),
        'Test': (test_df['tweet_length'].mean(), test_df['tweet_length'].std())
    }

    min_val = min(train_df['tweet_length'].min(), test_df['tweet_length'].min())
    max_val = max(train_df['tweet_length'].max(), test_df['tweet_length'].max())
    shared_bins = 20
    shared_range = (min_val, max_val)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)
    
    datasets = [('Train', train_df, '#4C72B0', axes[0]), 
                ('Test', test_df, '#DA70D6', axes[1])]

    for name, df, color, ax in datasets:
        sns.histplot(
            df['tweet_length'],
            bins=shared_bins,
            binrange=shared_range,
            stat='density', # Normalization
            kde=True,       
            color=color,
            alpha=0.6,
            ax=ax
        )
        
        ax.set_title(f"{name} Set")
        ax.set_xlabel("Word Count")
        ax.grid(visible=True, axis='y', alpha=0.3)
        
        # Add Stats Box to each subplot individually
        mean_val, std_val = stats[name]
        text_str = f"Mean: {mean_val:.2f}\nStd: {std_val:.2f}"
        
        ax.text(
            0.95, 0.95, text_str,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='#ccc')
        )

    axes[0].set_ylabel("Density")
    plt.suptitle('Train vs. Test Length Distribution', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()


def wordcloud_for_class(df, classes, label_int):
    label_name = classes[label_int]
    subset = df[df['label'] == label_int]['text'].astype(str)
    cleaned_texts = [clean_text_basic(t) for t in subset]
    all_words = [w for words in cleaned_texts for w in words]   # flatten
    text = " ".join(all_words)
    wc = WordCloud(width=600, height=400, background_color="white").generate(text)

    plt.figure(figsize=(8,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.title(f"WordCloud – {label_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()