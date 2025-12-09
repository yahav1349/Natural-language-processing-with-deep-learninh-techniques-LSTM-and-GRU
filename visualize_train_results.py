import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def print_predictions(df, num_examples=None, text_max_width=80):
    """
    Print predictions as a formatted table for misclassified examples.
    
    Args:
        df: DataFrame with columns 'text', 'gt_cls', 'pred_cls'
        num_examples: Number of examples to print (None = all)
        text_max_width: Maximum width for text column
    """
    # ANSI color codes
    RED = '\033[91m'
    GREEN = '\033[92m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Filter for misclassifications
    misclassified = df[df['gt_cls'] != df['pred_cls']]
    
    if num_examples is not None:
        misclassified = misclassified.head(num_examples)
    
    # Print title
    n = len(misclassified)
    print(f"{BOLD}{n} Examples of Misclassification{RESET}\n")
    
    # Calculate column widths
    gt_width = max(15, max(len(str(row['gt_cls'])) for _, row in misclassified.iterrows()) + 2)
    pred_width = max(15, max(len(str(row['pred_cls'])) for _, row in misclassified.iterrows()) + 2)
    
    # Print header
    print(f"{BOLD}{'Ground Truth':<{gt_width}} {'Prediction':<{pred_width}} {'Text'}{RESET}")
    print("-" * (gt_width + pred_width + text_max_width))
    
    # Print each example
    for idx, row in misclassified.iterrows():
        gt_cls = str(row['gt_cls'])
        pred_cls = str(row['pred_cls'])
        text = str(row['text'])
        
        # Truncate text if too long
        if len(text) > text_max_width:
            text = text[:text_max_width - 3] + "..."
        
        # Print row with colored values
        print(f"{GREEN}{gt_cls:<{gt_width}}{RESET} {RED}{pred_cls:<{pred_width}}{RESET} {text}")

def plot_training_progress(arr00, arr01, arr10, arr11, sup_tit, figsize=(14, 5), labels=[], ys=[]):
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(sup_tit, y=0.98, fontweight='bold')
    if len(labels) != 4:
        labels = ["Train Loss", "Validation Loss", "Train Accuracy", "Validation Accuracy"]
    if len(ys) != 2:
        ys = ["Cross Entropy Loss", "Accuracy"]
    ax[0].plot(arr00, label=labels[0], marker=".")
    ax[0].plot(arr01, label=labels[1], marker=".")
    ax[0].legend()
    ax[0].grid()
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel(ys[0], fontsize=16)
    
    
    ax[1].plot(arr10, label=labels[2], marker=".")
    ax[1].plot(arr11, label=labels[3], marker=".")
    ax[1].legend()
    ax[1].grid()
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel(ys[1], fontsize=16)
    plt.show()
    
def plot_confusion_matrix(test_df, results, classes, n_err_examples=10, figsize=(10, 8), cmap='Blues', title='Confusion matrix'):
    """
    Plot a confusion matrix heatmap from a DataFrame with ground truth and predicted classes.
    
    Args:
        df: DataFrame with 'gt_cls' and 'pred_cls' columns
        figsize: figure size
        cmap: colormap name
        normalize: if True, normalize values by row (True positive rate)
    
    Returns:
        matplotlib figure
    """
    
    df = test_df.copy()
    df["pred"] = results
    df["gt_cls"] = df.label.apply(lambda x: classes.get(x))
    df["pred_cls"] = df.pred.apply(lambda x: classes.get(x))
    
    # Compute confusion matrix
    cm = confusion_matrix(df['gt_cls'], df['pred_cls'])
    # Compute row-wise percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = cm / row_sums  # fraction from 0 to 1

    annot = cm.astype(str)

    for i in range(len(annot)):
        pct = cm_pct[i, i] * 100
        annot[i, i] = f"{cm[i, i]}\n ({pct:.1f}%)"
    # Get unique class labels
    classes = sorted(df['gt_cls'].unique())
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
    cm,
    annot=annot,
    fmt='',            
    cmap=cmap,
    xticklabels=classes,
    yticklabels=classes,
    cbar_kws={'label': 'Count'},
    linewidths=0.5,
    linecolor='gray',
    ax=ax
)
    
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ground Truth Class', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    err_examples = df[df.label != df.pred]
    samples = err_examples.sample(n=min(len(err_examples), n_err_examples), random_state=42)[["text", "gt_cls", "pred_cls"]]
    return samples