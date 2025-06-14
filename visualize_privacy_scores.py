import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paths
BASE_DIR    = "/data/cyu/IUIPC"
FILE_PATH   = f"{BASE_DIR}/results/privacy_attitudes_results.json"
OUTPUT_DIR  = f"{BASE_DIR}/results/visualizations"

# Load JSON data
def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Detailed horizontal bar chart
def create_detailed_bar_chart(data):
    models = [item['model'] for item in data]
    privacy_scores = [item.get('privacy_score', 0) for item in data]
    control_scores = [item['dimensions']['Control']['average'] for item in data]
    awareness_scores = [item['dimensions']['Awareness']['average'] for item in data]
    collection_scores = [item['dimensions']['Collection']['average'] for item in data]

    sorted_idx = np.argsort(models)
    models = [models[i] for i in sorted_idx]
    privacy_scores = [privacy_scores[i] for i in sorted_idx]
    control_scores = [control_scores[i] for i in sorted_idx]
    awareness_scores = [awareness_scores[i] for i in sorted_idx]
    collection_scores = [collection_scores[i] for i in sorted_idx]

    plt.figure(figsize=(12, max(8, len(models)*0.4)))
    ax = plt.gca()
    y_pos = np.arange(len(models))

    ax.barh(y_pos-0.3, control_scores, 0.2, label='Control', color='#8884d8')
    ax.barh(y_pos-0.1, awareness_scores, 0.2, label='Awareness', color='#82ca9d')
    ax.barh(y_pos+0.1, collection_scores, 0.2, label='Collection', color='#ffc658')
    ax.barh(y_pos+0.3, privacy_scores, 0.2, label='Privacy Score', color='#ff8042')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlim(0,7)
    ax.set_xlabel('Score (1-7 scale)')
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')
    ax.set_title('Privacy Dimensions by Model')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/detailed_bar_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Detailed bar chart saved to {OUTPUT_DIR}/detailed_bar_chart.png")

# Radar charts with neutral circle
def create_radar_charts(data):
    num = len(data)
    rows = (num + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4*rows), subplot_kw={'polar': True})
    axes = axes.flatten()

    dims = ['Control', 'Awareness', 'Collection']
    N = len(dims)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig_all = plt.figure(figsize=(10, 8))
    ax_all = fig_all.add_subplot(111, polar=True)
    cmap = plt.cm.viridis(np.linspace(0,1,num))

    # Plot each model
    for i, md in enumerate(data):
        scores = [md['dimensions'][d]['average'] for d in dims]
        scores += scores[:1]
        name = md['model'] if len(md['model']) <= 20 else md['model'][:17] + '...'
        color = cmap[i]
        if i < len(axes):
            ax = axes[i]
            ax.plot(angles, scores, linewidth=2, color=color)
            ax.fill(angles, scores, alpha=0.25, color=color)
            ax.set_xticks(angles[:-1]); ax.set_xticklabels(dims)
            ax.set_yticks([1,3,5,7]); ax.set_ylim(0,7)
            ax.set_title(name, size=10, y=1.1)
        ax_all.plot(angles, scores, linewidth=1.5, color=color, label=name)

    # Neutral circle at score=4
    neutral = [4] * len(angles)
    for ax in axes[:num]:
        ax.plot(angles, neutral, color='yellow', linewidth=1.5, linestyle='--')
    ax_all.plot(angles, neutral, color='yellow', linewidth=1.5, linestyle='--')

    for j in range(num, len(axes)): axes[j].axis('off')
    ax_all.set_xticks(angles[:-1]); ax_all.set_xticklabels(dims)
    ax_all.set_yticks([1,3,5,7]); ax_all.set_ylim(0,7)
    ax_all.set_title('Privacy Dimensions - All Models', size=14)
    fig_all.legend(loc='center left', bbox_to_anchor=(1.1,0.5))

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/individual_radar_charts.png", dpi=300, bbox_inches='tight')
    fig_all.tight_layout()
    fig_all.savefig(f"{OUTPUT_DIR}/consolidated_radar_chart.png", dpi=300, bbox_inches='tight')
    plt.close(fig); plt.close(fig_all)
    print(f"Radar charts saved to {OUTPUT_DIR}/individual_radar_charts.png")
    print(f"Consolidated radar chart saved to {OUTPUT_DIR}/consolidated_radar_chart.png")

# 2D scatter with right-aligned labels
def create_2d_privacy_plot(data):
    fig, ax = plt.subplots(figsize=(14, 10))
    markers = ['o','s','^','D','v','P','X','*','<','>','h','H','d','p']
    cmap = plt.cm.get_cmap('tab20', len(data))

    xs, ys, names = [], [], []
    sizes = []

    for i, md in enumerate(data):
        x = 8 - md['dimensions']['Collection']['average']
        y = md['dimensions']['Control']['average']
        name = md['model']
        size = 100 + (md.get('privacy_score',4.0) - 1) * 30
        xs.append(x); ys.append(y); names.append(name); sizes.append(size)
        color = cmap(i)
        marker = markers[i % len(markers)]
        ax.scatter(x, y, s=size, color=color, marker=marker)

    # Configure axes and grid
    ax.set_xlabel('Collection (reversed)')
    ax.set_ylabel('Control')
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0.5, 7.5)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axhline(4, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(4, color='gray', linestyle='-', alpha=0.5)

    # Quadrant labels
    quadrants = [
        ('High Control\nLow Collection Privacy', 1.5, 6.5),
        ('High Control\nHigh Collection Privacy', 6.5, 6.5),
        ('Low Control\nLow Collection Privacy', 1.5, 1.5),
        ('Low Control\nHigh Collection Privacy', 6.5, 1.5)
    ]
    for text, xq, yq in quadrants:
        ax.text(xq, yq, text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    # Compute evenly spaced label positions on right
    y_min, y_max = ax.get_ylim()
    label_ys = np.linspace(y_max, y_min, len(data))
    orig_xlim = ax.get_xlim()
    x_label = orig_xlim[1] + (orig_xlim[1] - orig_xlim[0]) * 0.05
    ax.set_xlim(orig_xlim[0], x_label + 1.0)

    # Draw leader lines and place labels
    for i, name in enumerate(names):
        ax.plot([xs[i], x_label], [ys[i], label_ys[i]], color=cmap(i), linewidth=0.5)
        ax.text(x_label + 0.1, label_ys[i], name, va='center', fontsize=10, color=cmap(i))

    plt.title('Privacy: Collection vs Control', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2d_privacy_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"2D privacy plot saved to {OUTPUT_DIR}/2d_privacy_plot.png")

# Main execution
def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = load_data(FILE_PATH)
    create_detailed_bar_chart(data)
    create_radar_charts(data)
    create_2d_privacy_plot(data)
    print("All visualizations completed successfully!")

if __name__ == '__main__':
    main()
