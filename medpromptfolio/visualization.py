"""
Heterogeneity Visualization Utilities for Federated Learning

Provides tools to visualize and analyze data heterogeneity across
virtual clients, which is essential for publication-quality figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple
import os


def plot_label_distribution_heatmap(
    label_distributions: np.ndarray,
    client_names: List[str],
    class_names: List[str],
    title: str = "Label Distribution Across Clients",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "YlOrRd",
):
    """
    Plot heatmap of label distributions across clients.

    Args:
        label_distributions: Array of shape (num_clients, num_classes) with prevalences
        client_names: Names of each client
        class_names: Names of each class/pathology
        title: Plot title
        save_path: If provided, save figure to this path
        figsize: Figure size
        cmap: Colormap name
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(label_distributions, aspect='auto', cmap=cmap)

    # Set ticks
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(client_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(client_names)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Prevalence", rotation=-90, va="bottom")

    # Add value annotations
    for i in range(len(client_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, f"{label_distributions[i, j]:.2f}",
                           ha="center", va="center", color="black", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("Pathology")
    ax.set_ylabel("Client")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def plot_sample_distribution_bar(
    samples_per_client: List[int],
    client_names: List[str],
    dataset_colors: Optional[Dict[str, str]] = None,
    title: str = "Sample Distribution Across Clients",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
):
    """
    Plot bar chart of sample counts per client.

    Args:
        samples_per_client: Number of samples per client
        client_names: Names of each client
        dataset_colors: Optional dict mapping dataset prefix to color
        title: Plot title
        save_path: If provided, save figure to this path
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Assign colors based on dataset
    if dataset_colors is None:
        dataset_colors = {
            'nih': '#1f77b4',
            'chexpert': '#ff7f0e',
            'mimic': '#2ca02c',
            'vindr': '#d62728',
            'synthetic': '#9467bd',
        }

    colors = []
    for name in client_names:
        color = '#7f7f7f'  # default gray
        for prefix, c in dataset_colors.items():
            if prefix.lower() in name.lower():
                color = c
                break
        colors.append(color)

    bars = ax.bar(range(len(client_names)), samples_per_client, color=colors)

    ax.set_xticks(range(len(client_names)))
    ax.set_xticklabels(client_names, rotation=45, ha='right')
    ax.set_xlabel("Client")
    ax.set_ylabel("Number of Samples")
    ax.set_title(title)

    # Add legend
    legend_handles = []
    for prefix, color in dataset_colors.items():
        if any(prefix.lower() in name.lower() for name in client_names):
            legend_handles.append(mpatches.Patch(color=color, label=prefix.upper()))
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def plot_heterogeneity_comparison(
    stats_by_alpha: Dict[float, Dict],
    class_names: List[str],
    title: str = "Effect of Dirichlet Alpha on Heterogeneity",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
):
    """
    Compare label distributions under different alpha values.

    Args:
        stats_by_alpha: Dict mapping alpha -> heterogeneity stats
        class_names: Names of classes
        title: Plot title
        save_path: If provided, save figure to this path
        figsize: Figure size
    """
    alphas = sorted(stats_by_alpha.keys())
    n_alphas = len(alphas)

    fig, axes = plt.subplots(1, n_alphas, figsize=figsize, sharey=True)
    if n_alphas == 1:
        axes = [axes]

    for idx, alpha in enumerate(alphas):
        stats = stats_by_alpha[alpha]
        dists = np.array(stats['label_distributions'])

        ax = axes[idx]
        im = ax.imshow(dists, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)

        ax.set_title(f"α = {alpha}")
        ax.set_xlabel("Pathology")
        if idx == 0:
            ax.set_ylabel("Client")

        ax.set_xticks(np.arange(len(class_names)))
        ax.set_xticklabels([c[:8] for c in class_names], rotation=45, ha='right', fontsize=8)

    # Add single colorbar
    fig.colorbar(im, ax=axes, shrink=0.6, label='Prevalence')

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, axes


def plot_js_divergence_matrix(
    label_distributions: np.ndarray,
    client_names: List[str],
    title: str = "Jensen-Shannon Divergence Between Clients",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
):
    """
    Plot pairwise JS divergence between clients.

    Args:
        label_distributions: Array of shape (num_clients, num_classes)
        client_names: Names of each client
        title: Plot title
        save_path: If provided, save figure to this path
        figsize: Figure size
    """
    from scipy.spatial.distance import jensenshannon

    n_clients = len(client_names)
    js_matrix = np.zeros((n_clients, n_clients))

    for i in range(n_clients):
        for j in range(n_clients):
            if i != j:
                # Add small epsilon to avoid issues with zero probabilities
                p = label_distributions[i] + 1e-10
                q = label_distributions[j] + 1e-10
                p = p / p.sum()
                q = q / q.sum()
                js_matrix[i, j] = jensenshannon(p, q)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(js_matrix, cmap='Blues')

    ax.set_xticks(np.arange(n_clients))
    ax.set_yticks(np.arange(n_clients))
    ax.set_xticklabels(client_names, rotation=45, ha='right')
    ax.set_yticklabels(client_names)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("JS Divergence", rotation=-90, va="bottom")

    # Add annotations
    for i in range(n_clients):
        for j in range(n_clients):
            text = ax.text(j, i, f"{js_matrix[i, j]:.2f}",
                           ha="center", va="center", color="black", fontsize=8)

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def plot_class_distribution_radar(
    label_distributions: np.ndarray,
    client_names: List[str],
    class_names: List[str],
    title: str = "Class Distribution Radar Plot",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
):
    """
    Plot radar chart comparing class distributions across clients.

    Args:
        label_distributions: Array of shape (num_clients, num_classes)
        client_names: Names of each client (max 6 for visibility)
        class_names: Names of classes
        title: Plot title
        save_path: If provided, save figure to this path
        figsize: Figure size
    """
    num_vars = len(class_names)

    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # Colors for different clients
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(client_names), 10)))

    for idx, (name, dist) in enumerate(zip(client_names[:6], label_distributions[:6])):
        values = dist.tolist()
        values += values[:1]  # Complete the loop

        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(class_names)
    ax.set_title(title)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def create_heterogeneity_report(
    fed_dataset,
    output_dir: str,
    prefix: str = "heterogeneity",
):
    """
    Generate comprehensive heterogeneity report with all visualizations.

    Args:
        fed_dataset: FederatedChestXrayDataset or FederatedVirtualClients instance
        output_dir: Directory to save figures
        prefix: Filename prefix for saved figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get data
    client_names = []
    samples_per_client = []
    label_distributions = []

    for client_id in range(fed_dataset.num_clients):
        if hasattr(fed_dataset, 'get_client_name'):
            client_names.append(fed_dataset.get_client_name(client_id))
        else:
            client_names.append(f"Client_{client_id}")

        if hasattr(fed_dataset, 'get_num_samples'):
            samples_per_client.append(fed_dataset.get_num_samples(client_id))
        else:
            samples_per_client.append(len(fed_dataset.get_client_dataset(client_id)))

        # Get label distribution
        if hasattr(fed_dataset, 'get_label_distribution'):
            dist = fed_dataset.get_label_distribution(client_id)
        else:
            dataset = fed_dataset.get_client_dataset(client_id)
            if hasattr(dataset, 'labels'):
                labels = np.array(dataset.labels)
                dist = labels.mean(axis=0) if len(labels.shape) > 1 else None
            else:
                dist = None

        if dist is not None:
            label_distributions.append(dist)

    label_distributions = np.array(label_distributions)

    # Get class names
    if hasattr(fed_dataset, 'classnames') and fed_dataset.classnames:
        class_names = fed_dataset.classnames
    else:
        class_names = [f"Class_{i}" for i in range(label_distributions.shape[1])]

    print(f"\nGenerating heterogeneity report in {output_dir}")
    print("=" * 60)

    # 1. Label distribution heatmap
    plot_label_distribution_heatmap(
        label_distributions,
        client_names,
        class_names,
        save_path=os.path.join(output_dir, f"{prefix}_label_heatmap.png")
    )
    plt.close()

    # 2. Sample distribution bar chart
    plot_sample_distribution_bar(
        samples_per_client,
        client_names,
        save_path=os.path.join(output_dir, f"{prefix}_sample_distribution.png")
    )
    plt.close()

    # 3. JS divergence matrix
    if len(label_distributions) > 0:
        plot_js_divergence_matrix(
            label_distributions,
            client_names,
            save_path=os.path.join(output_dir, f"{prefix}_js_divergence.png")
        )
        plt.close()

    # 4. Radar plot
    if len(label_distributions) > 0 and len(client_names) <= 8:
        plot_class_distribution_radar(
            label_distributions,
            client_names,
            class_names,
            save_path=os.path.join(output_dir, f"{prefix}_radar.png")
        )
        plt.close()

    # 5. Save statistics to text file
    stats_path = os.path.join(output_dir, f"{prefix}_stats.txt")
    with open(stats_path, 'w') as f:
        f.write("Heterogeneity Statistics\n")
        f.write("=" * 60 + "\n\n")

        f.write("Sample Distribution:\n")
        f.write(f"  Total samples: {sum(samples_per_client)}\n")
        f.write(f"  Mean per client: {np.mean(samples_per_client):.1f}\n")
        f.write(f"  Std per client: {np.std(samples_per_client):.1f}\n")
        f.write(f"  Min: {min(samples_per_client)}\n")
        f.write(f"  Max: {max(samples_per_client)}\n\n")

        f.write("Label Distribution:\n")
        if len(label_distributions) > 0:
            mean_dist = label_distributions.mean(axis=0)
            std_dist = label_distributions.std(axis=0)
            f.write(f"  Mean prevalence: {mean_dist}\n")
            f.write(f"  Std prevalence: {std_dist}\n\n")

            # Compute average JS divergence
            from scipy.spatial.distance import jensenshannon
            js_divs = []
            for i in range(len(label_distributions)):
                for j in range(i + 1, len(label_distributions)):
                    p = label_distributions[i] + 1e-10
                    q = label_distributions[j] + 1e-10
                    js_divs.append(jensenshannon(p / p.sum(), q / q.sum()))
            f.write(f"  Avg JS divergence: {np.mean(js_divs):.4f}\n")

        f.write("\nPer-Client Details:\n")
        for i, name in enumerate(client_names):
            f.write(f"  {name}: {samples_per_client[i]} samples\n")
            if i < len(label_distributions):
                dist_str = ", ".join([f"{v:.3f}" for v in label_distributions[i]])
                f.write(f"    Distribution: [{dist_str}]\n")

    print(f"Saved statistics: {stats_path}")
    print("=" * 60)
    print("Report generation complete!")

    return {
        'client_names': client_names,
        'samples_per_client': samples_per_client,
        'label_distributions': label_distributions,
        'class_names': class_names,
    }
