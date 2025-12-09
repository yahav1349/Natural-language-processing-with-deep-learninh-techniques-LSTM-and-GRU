import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_params_barplot(data, figsize=(16, 8), top_n=None, sort_by_score=True):
    """
    Bar plot where each bar represents a parameter configuration.

    Args:
        data: CV results data
        figsize: figure size
        top_n: number of top configurations to show (None = show all)
        sort_by_score: whether to sort bars by mean score

    Returns:
        matplotlib figure
    """
    df = pd.DataFrame([
        {**r['params'], 'mean_score': r['mean_score'], 'std_score': r['std_score']}
        for r in data['all_results']
    ])

    # Sort by score if requested
    if sort_by_score:
        df = df.sort_values('mean_score', ascending=False)

    # Limit to top N if specified
    if top_n is not None:
        df = df.head(top_n)

    # Create labels from parameters
    labels = []
    # Create labels from parameters
    param_cols = [col for col in df.columns if col not in ['mean_score', 'std_score']]
    labels = []
    for idx, row in df.iterrows():
      # Format as embed_dim/hidden_dim/num_epochs
      param_str = '/'.join([str(int(row[col])) for col in param_cols])
      labels.append(param_str)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(df))
    bars = ax.bar(x, df['mean_score'], yerr=df['std_score'],
                   capsize=4, alpha=0.7, edgecolor='black', linewidth=1.2, color='orchid')

    # Highlight best configuration
    if len(df) > 0:
        best_idx = 0 if sort_by_score else df['mean_score'].idxmax()
        if sort_by_score:
            bars[0].set_edgecolor('red')
            bars[0].set_linewidth(3)
        else:
            best_position = df.index.get_loc(best_idx)
            bars[best_position].set_edgecolor('red')
            bars[best_position].set_linewidth(3)

    # Add value labels on top of bars
    for i, (idx, row) in enumerate(df.iterrows()):
        height = row['mean_score']
        ax.text(i, height, f'{height:.4f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
        break

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')

    # Create axis label from parameter names
    xlabel = ' / '.join(param_cols)
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')

    title = f'Accuracy by Configuration'
    if top_n:
        title += f' (Top {top_n})'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(df['mean_score'].min() - 0.01, df['mean_score'].max() + 0.015)

    plt.tight_layout()
    return fig


def plot_mean_scores_comparison(data, figsize=(16, 6)):
    """
    Side-by-side bar plots showing mean scores for embedding dimension and hidden dimension,
    with shared y-axis and colorbar.

    Args:
        data: CV results data
        figsize: figure size

    Returns:
        matplotlib figure
    """
    df = pd.DataFrame([
        {**r['params'], 'mean_score': r['mean_score'], 'std_score': r['std_score']}
        for r in data['all_results']
    ])

    # Group by embed_dim and hidden_dim
    grouped_embed = df.groupby('embed_dim').agg({
        'mean_score': ['mean', 'std', 'count'],
        'std_score': 'mean'
    })

    grouped_hidden = df.groupby('hidden_dim').agg({
        'mean_score': ['mean', 'std', 'count'],
        'std_score': 'mean'
    })

    # Create figure with two subplots sharing y-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Get global min/max for shared colormap
    all_means = np.concatenate([
        grouped_embed['mean_score']['mean'].values,
        grouped_hidden['mean_score']['mean'].values
    ])
    vmin, vmax = all_means.min(), all_means.max()

    # Plot 1: Embedding Dimension
    x1 = np.arange(len(grouped_embed))
    embed_dims = grouped_embed.index
    means1 = grouped_embed['mean_score']['mean']
    stds1 = grouped_embed['mean_score']['std']

    bars1 = ax1.bar(x1, means1, yerr=stds1, capsize=5, alpha=0.7,
                    edgecolor='black', linewidth=1.5)

    # Highlight best for embed_dim
    best_idx1 = means1.idxmax()
    best_pos1 = list(grouped_embed.index).index(best_idx1)
    bars1[best_pos1].set_edgecolor('red')
    bars1[best_pos1].set_linewidth(3)

    # Add value labels
    for i, mean_val in enumerate(means1):
        ax1.text(i, mean_val, f'{mean_val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add "BEST" marker
    ax1.text(best_pos1, means1.iloc[best_pos1] * 0.98, 'BEST',
            ha='center', va='top', fontsize=9, fontweight='bold',
            color='red', bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='yellow', alpha=0.7))

    ax1.set_xticks(x1)
    ax1.set_xticklabels(embed_dims, fontsize=11)
    ax1.set_xlabel('Embedding Dimension', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (averaged over other params)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy by Embedding Dimension', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 2: Hidden Dimension
    x2 = np.arange(len(grouped_hidden))
    hidden_dims = grouped_hidden.index
    means2 = grouped_hidden['mean_score']['mean']
    stds2 = grouped_hidden['mean_score']['std']

    bars2 = ax2.bar(x2, means2, yerr=stds2, capsize=5, alpha=0.7,
                    edgecolor='black', linewidth=1.5, color='orchid')
    # Highlight best for hidden_dim
    best_idx2 = means2.idxmax()
    best_pos2 = list(grouped_hidden.index).index(best_idx2)
    bars2[best_pos2].set_edgecolor('red')
    bars2[best_pos2].set_linewidth(3)

    # Add value labels
    for i, mean_val in enumerate(means2):
        ax2.text(i, mean_val, f'{mean_val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add "BEST" marker
    ax2.text(best_pos2, means2.iloc[best_pos2] * 0.98, 'BEST',
            ha='center', va='top', fontsize=9, fontweight='bold',
            color='red', bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='yellow', alpha=0.7))

    ax2.set_xticks(x2)
    ax2.set_xticklabels(hidden_dims, fontsize=11)
    ax2.set_xlabel('Hidden Dimension', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy by Hidden Dimension', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_scatter_with_third_param(data, param1='embed_dim', param2='hidden_dim',
                                   param3='num_epochs', figsize=(12, 8),
                                   cmap='viridis', select_best_per_group=True):
    """
    Scatter plot with two parameters on axes, third parameter as marker shape,
    and mean score as color.

    Args:
        data: CV results data
        param1: parameter for x-axis
        param2: parameter for y-axis
        param3: parameter for marker shape (typically num_epochs)
        figsize: figure size
        cmap: colormap name
        select_best_per_group: if True, only plot the best param3 value for each (param1, param2) combination

    Returns:
        matplotlib figure
    """
    df = pd.DataFrame([
        {**r['params'], 'mean_score': r['mean_score'], 'std_score': r['std_score']}
        for r in data['all_results']
    ])

    # Select only the best result for each (param1, param2) combination
    if select_best_per_group:
        df = df.loc[df.groupby([param1, param2])['mean_score'].idxmax()].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Define marker shapes for different values of param3
    marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    unique_param3 = sorted(df[param3].unique())
    param3_to_marker = {val: marker_styles[i % len(marker_styles)] for i, val in enumerate(unique_param3)}

    # Get min/max scores for colormap normalization
    min_score = df['mean_score'].min()
    max_score = df['mean_score'].max()

    # Plot each group with different marker shapes
    legend_elements = []
    for param3_val in unique_param3:
        subset = df[df[param3] == param3_val]
        marker = param3_to_marker[param3_val]

        scatter = ax.scatter(subset[param1], subset[param2],
                            c=subset['mean_score'],
                            cmap=cmap,
                            vmin=min_score,
                            vmax=max_score,
                            marker=marker,
                            s=200,
                            alpha=0.7,
                            edgecolors='black',
                            linewidths=1.5)

        # Create legend element for this marker
        legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w',
                                         markerfacecolor='gray', markersize=10,
                                         label=f'{param3}={int(param3_val)}',
                                         markeredgecolor='black', markeredgewidth=1.5))

    # Add colorbar for mean_score with more padding to the right
    cbar = plt.colorbar(scatter, ax=ax, label='Accuracy', pad=0.008)

    # Annotate best configuration
    best_idx = df['mean_score'].idxmax()
    best_marker = param3_to_marker[df.loc[best_idx, param3]]
    ax.scatter(df.loc[best_idx, param1], df.loc[best_idx, param2],
              s=300,
              c='red',
              marker='*',
              edgecolors='gold',
              linewidths=3,
              label='Best Config',
              zorder=10)

    # Add text annotation for best
    ax.annotate(f"Best: {df.loc[best_idx, 'mean_score']:.4f}",
               (df.loc[best_idx, param1], df.loc[best_idx, param2]),
               xytext=(-80, 10), textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
               fontsize=10, fontweight='bold',
               zorder=11)

    # Add legend for marker shapes outside the plot frame
    ax.legend(handles=legend_elements,
             title=f'{param3} (marker shape)',
             loc='upper left',
             bbox_to_anchor=(0.01, 0.8),
             frameon=True,
             fontsize=9)

    ax.set_xlabel(param1.replace("_", " ").title(), fontsize=12, fontweight='bold')
    ax.set_ylabel(param2.replace("_", " ").title(), fontsize=12, fontweight='bold')
    ax.set_title(f'{param1.replace("_", " ").title()} vs {param2.replace("_", " ").title()}\n(Marker shape: {param3.replace("_", " ").title()}, Color: Accuracy)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig