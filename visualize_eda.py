import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
# from eda_functions import clean_text_basic
from eda_functions import clean_text_basic


def plot_class_distribution(df, classes, title):
    # Map integer labels to strings
    df['label_name'] = df['label'].map(classes)

    # Define a "represtive" color for each class
    label_colors = {
        "Sadness": 'grey',
        "Joy": '#FFF530',
        "Love": 'pink',
        "Anger": 'maroon',
        "Fear": '#4B0082',
        "Surprise": 'orange'
    }

    plt.figure(figsize=(8,5))
    ax = sns.countplot(
        x='label_name',
        hue='label_name',     
        data=df,
        palette=label_colors,
        order=df['label_name'].value_counts().index,
        dodge=False,  
        legend=False          
    )

    plt.title(f"{title} Class distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()



def tweet_length_distribution(df, title):
    df['tweet_length'] = df['text'].apply(lambda x: len(x.split()))
    mean_len = df['tweet_length'].mean()
    std_len = df['tweet_length'].std()

    plt.figure(figsize=(8,5))
    ax = sns.histplot(df['tweet_length'], bins=20, kde=False, color='orchid')

    plt.title(f"{title} Tweet length distribution (words)")
    plt.xlabel("Word count")
    plt.ylabel("Frequency")

    # mean and std in top-right corner
    plt.text(
        0.95, 0.95,
        f"Mean: {mean_len:.2f}\nStd: {std_len:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.5)
    )
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