import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_class_distribution(train_df, test_df, figsize=(14, 5)):
    """
    Plot class distribution for train and test sets with imbalance analysis.
    
    Args:
        train_df: DataFrame with 'gt_cls' column
        test_df: DataFrame with 'gt_cls' column
        figsize: Figure size
    
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Count classes
    train_counts = train_df['gt_cls'].value_counts().sort_index()
    test_counts = test_df['gt_cls'].value_counts().sort_index()
    
    # Calculate percentages
    train_pct = (train_counts / len(train_df) * 100).round(2)
    test_pct = (test_counts / len(test_df) * 100).round(2)
    
    # Plot 1: Train distribution
    ax1 = axes[0]
    bars1 = ax1.bar(range(len(train_counts)), train_counts.values, 
                    color=plt.cm.viridis(np.linspace(0, 1, len(train_counts))),
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_xticks(range(len(train_counts)))
    ax1.set_xticklabels(train_counts.index, rotation=45, ha='right')
    ax1.set_xlabel('Emotion Class', fontweight='bold')
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title(f'Train Set Distribution (n={len(train_df)})', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add count and percentage labels
    for i, (count, pct) in enumerate(zip(train_counts.values, train_pct.values)):
        ax1.text(i, count, f'{count}\n({pct}%)', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Test distribution
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(test_counts)), test_counts.values,
                    color=plt.cm.viridis(np.linspace(0, 1, len(test_counts))),
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_xticks(range(len(test_counts)))
    ax2.set_xticklabels(test_counts.index, rotation=45, ha='right')
    ax2.set_xlabel('Emotion Class', fontweight='bold')
    ax2.set_ylabel('Count', fontweight='bold')
    ax2.set_title(f'Test Set Distribution (n={len(test_df)})', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add count and percentage labels
    for i, (count, pct) in enumerate(zip(test_counts.values, test_pct.values)):
        ax2.text(i, count, f'{count}\n({pct}%)', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 3: Imbalance ratio
    ax3 = axes[2]
    
    # Calculate imbalance ratio (max/min)
    train_imbalance = train_counts.max() / train_counts.min()
    test_imbalance = test_counts.max() / test_counts.min()
    
    # Normalize counts for comparison
    train_norm = train_counts / train_counts.max()
    test_norm = test_counts / test_counts.max()
    
    x = np.arange(len(train_counts))
    width = 0.35
    
    bars_train = ax3.bar(x - width/2, train_norm.values, width, 
                         label='Train', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars_test = ax3.bar(x + width/2, test_norm.values, width,
                        label='Test', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(train_counts.index, rotation=45, ha='right')
    ax3.set_xlabel('Emotion Class', fontweight='bold')
    ax3.set_ylabel('Normalized Count (max=1)', fontweight='bold')
    ax3.set_title(f'Class Balance Comparison\nTrain IR: {train_imbalance:.2f}, Test IR: {test_imbalance:.2f}', 
                  fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    
    plt.tight_layout()
    return fig


def analyze_class_imbalance(train_df, test_df):
    """
    Print detailed class imbalance statistics.
    
    Args:
        train_df: DataFrame with 'gt_cls' column
        test_df: DataFrame with 'gt_cls' column
    """
    BOLD = '\033[1m'
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    
    print(f"{BOLD}=== Class Imbalance Analysis ==={RESET}\n")
    
    # Train set analysis
    train_counts = train_df['gt_cls'].value_counts().sort_index()
    train_pct = (train_counts / len(train_df) * 100)
    
    print(f"{BOLD}Train Set:{RESET}")
    print(f"  Total samples: {len(train_df)}")
    print(f"  Number of classes: {len(train_counts)}")
    print(f"  Majority class: {train_counts.idxmax()} ({train_counts.max()} samples, {train_pct.max():.2f}%)")
    print(f"  Minority class: {train_counts.idxmin()} ({train_counts.min()} samples, {train_pct.min():.2f}%)")
    
    imbalance_ratio_train = train_counts.max() / train_counts.min()
    print(f"  Imbalance Ratio (max/min): {imbalance_ratio_train:.2f}")
    
    if imbalance_ratio_train > 3:
        print(f"  {RED}⚠ High imbalance detected!{RESET}")
    elif imbalance_ratio_train > 1.5:
        print(f"  {YELLOW}⚠ Moderate imbalance detected{RESET}")
    else:
        print(f"  {GREEN}✓ Relatively balanced{RESET}")
    
    print()
    
    # Test set analysis
    test_counts = test_df['gt_cls'].value_counts().sort_index()
    test_pct = (test_counts / len(test_df) * 100)
    
    print(f"{BOLD}Test Set:{RESET}")
    print(f"  Total samples: {len(test_df)}")
    print(f"  Number of classes: {len(test_counts)}")
    print(f"  Majority class: {test_counts.idxmax()} ({test_counts.max()} samples, {test_pct.max():.2f}%)")
    print(f"  Minority class: {test_counts.idxmin()} ({test_counts.min()} samples, {test_pct.min():.2f}%)")
    
    imbalance_ratio_test = test_counts.max() / test_counts.min()
    print(f"  Imbalance Ratio (max/min): {imbalance_ratio_test:.2f}")
    
    if imbalance_ratio_test > 3:
        print(f"  {RED}⚠ High imbalance detected!{RESET}")
    elif imbalance_ratio_test > 1.5:
        print(f"  {YELLOW}⚠ Moderate imbalance detected{RESET}")
    else:
        print(f"  {GREEN}✓ Relatively balanced{RESET}")
    
    print()
    
    # Per-class breakdown
    print(f"{BOLD}Per-Class Distribution:{RESET}")
    print(f"{'Class':<15} {'Train Count':<15} {'Train %':<12} {'Test Count':<15} {'Test %':<12}")
    print("-" * 70)
    
    for cls in sorted(train_counts.index):
        train_c = train_counts.get(cls, 0)
        train_p = train_pct.get(cls, 0)
        test_c = test_counts.get(cls, 0)
        test_p = test_pct.get(cls, 0)
        print(f"{str(cls):<15} {train_c:<15} {train_p:<12.2f} {test_c:<15} {test_p:<12.2f}")


def plot_text_length_analysis(train_df, test_df, figsize=(16, 10)):
    """
    Analyze and plot text length distributions.
    
    Args:
        train_df: DataFrame with 'text' and 'gt_cls' columns
        test_df: DataFrame with 'text' and 'gt_cls' columns
        figsize: Figure size
    
    Returns:
        matplotlib figure
    """
    # Calculate text lengths
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df['text_len_chars'] = train_df['text'].str.len()
    train_df['text_len_words'] = train_df['text'].str.split().str.len()
    
    test_df['text_len_chars'] = test_df['text'].str.len()
    test_df['text_len_words'] = test_df['text'].str.split().str.len()
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Plot 1: Character length distribution - Train
    ax = axes[0, 0]
    ax.hist(train_df['text_len_chars'], bins=50, alpha=0.7, color='skyblue', 
            edgecolor='black', linewidth=1.2)
    ax.axvline(train_df['text_len_chars'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f"Mean: {train_df['text_len_chars'].mean():.1f}")
    ax.axvline(train_df['text_len_chars'].median(), color='green', linestyle='--', 
               linewidth=2, label=f"Median: {train_df['text_len_chars'].median():.1f}")
    ax.set_xlabel('Text Length (characters)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Train: Character Length Distribution', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Character length distribution - Test
    ax = axes[0, 1]
    ax.hist(test_df['text_len_chars'], bins=50, alpha=0.7, color='lightcoral', 
            edgecolor='black', linewidth=1.2)
    ax.axvline(test_df['text_len_chars'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f"Mean: {test_df['text_len_chars'].mean():.1f}")
    ax.axvline(test_df['text_len_chars'].median(), color='green', linestyle='--', 
               linewidth=2, label=f"Median: {test_df['text_len_chars'].median():.1f}")
    ax.set_xlabel('Text Length (characters)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Test: Character Length Distribution', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Word length distribution - Train
    ax = axes[0, 2]
    ax.hist(train_df['text_len_words'], bins=50, alpha=0.7, color='lightgreen', 
            edgecolor='black', linewidth=1.2)
    ax.axvline(train_df['text_len_words'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f"Mean: {train_df['text_len_words'].mean():.1f}")
    ax.axvline(train_df['text_len_words'].median(), color='green', linestyle='--', 
               linewidth=2, label=f"Median: {train_df['text_len_words'].median():.1f}")
    ax.set_xlabel('Text Length (words)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Train: Word Count Distribution', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Box plot by class - Train (characters)
    ax = axes[1, 0]
    classes = sorted(train_df['gt_cls'].unique())
    data_chars = [train_df[train_df['gt_cls'] == cls]['text_len_chars'].values for cls in classes]
    bp = ax.boxplot(data_chars, labels=classes, patch_artist=True)
    for patch, color in zip(bp['boxes'], plt.cm.viridis(np.linspace(0, 1, len(classes)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel('Emotion Class', fontweight='bold')
    ax.set_ylabel('Text Length (characters)', fontweight='bold')
    ax.set_title('Train: Character Length by Class', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 5: Box plot by class - Test (characters)
    ax = axes[1, 1]
    data_chars_test = [test_df[test_df['gt_cls'] == cls]['text_len_chars'].values for cls in classes]
    bp = ax.boxplot(data_chars_test, labels=classes, patch_artist=True)
    for patch, color in zip(bp['boxes'], plt.cm.viridis(np.linspace(0, 1, len(classes)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel('Emotion Class', fontweight='bold')
    ax.set_ylabel('Text Length (characters)', fontweight='bold')
    ax.set_title('Test: Character Length by Class', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 6: Box plot by class - Train (words)
    ax = axes[1, 2]
    data_words = [train_df[train_df['gt_cls'] == cls]['text_len_words'].values for cls in classes]
    bp = ax.boxplot(data_words, labels=classes, patch_artist=True)
    for patch, color in zip(bp['boxes'], plt.cm.viridis(np.linspace(0, 1, len(classes)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel('Emotion Class', fontweight='bold')
    ax.set_ylabel('Text Length (words)', fontweight='bold')
    ax.set_title('Train: Word Count by Class', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def analyze_text_length_stats(train_df, test_df):
    """
    Print detailed text length statistics.
    
    Args:
        train_df: DataFrame with 'text' and 'gt_cls' columns
        test_df: DataFrame with 'text' and 'gt_cls' columns
    """
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    # Calculate lengths
    train_chars = train_df['text'].str.len()
    train_words = train_df['text'].str.split().str.len()
    test_chars = test_df['text'].str.len()
    test_words = test_df['text'].str.split().str.len()
    
    print(f"{BOLD}=== Text Length Statistics ==={RESET}\n")
    
    print(f"{BOLD}Train Set:{RESET}")
    print(f"  Character length:")
    print(f"    Mean:   {train_chars.mean():.2f}")
    print(f"    Median: {train_chars.median():.2f}")
    print(f"    Std:    {train_chars.std():.2f}")
    print(f"    Min:    {train_chars.min()}")
    print(f"    Max:    {train_chars.max()}")
    print(f"  Word count:")
    print(f"    Mean:   {train_words.mean():.2f}")
    print(f"    Median: {train_words.median():.2f}")
    print(f"    Std:    {train_words.std():.2f}")
    print(f"    Min:    {train_words.min()}")
    print(f"    Max:    {train_words.max()}")
    
    print(f"\n{BOLD}Test Set:{RESET}")
    print(f"  Character length:")
    print(f"    Mean:   {test_chars.mean():.2f}")
    print(f"    Median: {test_chars.median():.2f}")
    print(f"    Std:    {test_chars.std():.2f}")
    print(f"    Min:    {test_chars.min()}")
    print(f"    Max:    {test_chars.max()}")
    print(f"  Word count:")
    print(f"    Mean:   {test_words.mean():.2f}")
    print(f"    Median: {test_words.median():.2f}")
    print(f"    Std:    {test_words.std():.2f}")
    print(f"    Min:    {test_words.min()}")
    print(f"    Max:    {test_words.max()}")
    
    print(f"\n{BOLD}Per-Class Statistics (Train):{RESET}")
    print(f"{'Class':<15} {'Avg Chars':<12} {'Avg Words':<12} {'Std Chars':<12} {'Std Words':<12}")
    print("-" * 65)
    
    for cls in sorted(train_df['gt_cls'].unique()):
        subset = train_df[train_df['gt_cls'] == cls]
        avg_chars = subset['text'].str.len().mean()
        avg_words = subset['text'].str.split().str.len().mean()
        std_chars = subset['text'].str.len().std()
        std_words = subset['text'].str.split().str.len().std()
        print(f"{str(cls):<15} {avg_chars:<12.2f} {avg_words:<12.2f} {std_chars:<12.2f} {std_words:<12.2f}")
