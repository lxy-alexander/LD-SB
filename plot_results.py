"""
Plot LD-SB experiment results from JSON output files.
Generates figures for Rich vs Lazy comparison across network depths.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def load_results(output_dir="outputs"):
    """Load all results JSON files, organized by regime and layers."""
    results = {"rich": {}, "lazy": {}}
    output_path = Path(output_dir)
    
    # Load regime-specific files (new format)
    for regime in ["rich", "lazy"]:
        for json_file in sorted(output_path.glob(f"results_{regime}_layer*.json")):
            with open(json_file) as f:
                data = json.load(f)
                num_layers = data["config"]["num_layers"]
                results[regime][num_layers] = data
    
    # Fallback: load old format files as "rich"
    if not results["rich"]:
        for json_file in sorted(output_path.glob("results_layer*.json")):
            with open(json_file) as f:
                data = json.load(f)
                num_layers = data["config"]["num_layers"]
                results["rich"][num_layers] = data
    
    return results


def plot_rich_vs_lazy_comparison(results, save_dir="results"):
    """Generate comparison plots for Rich vs Lazy regimes."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    rich = results["rich"]
    lazy = results["lazy"]
    
    if not rich or not lazy:
        print("Need both Rich and Lazy results for comparison plots")
        return
    
    layers = sorted(set(rich.keys()) & set(lazy.keys()))
    
    # Extract metrics for both regimes
    rich_ranks = [rich[l]["final_effective_rank"] for l in layers]
    lazy_ranks = [lazy[l]["final_effective_rank"] for l in layers]
    
    rich_val_acc = [rich[l]["final_val_acc"] for l in layers]
    lazy_val_acc = [lazy[l]["final_val_acc"] for l in layers]
    
    rich_p_perp = [rich[l]["ldsb_metrics"]["P_perp_pC"] for l in layers]
    lazy_p_perp = [lazy[l]["ldsb_metrics"]["P_perp_pC"] for l in layers]
    
    rich_p_pc = [rich[l]["ldsb_metrics"]["P_pC"] for l in layers]
    lazy_p_pc = [lazy[l]["ldsb_metrics"]["P_pC"] for l in layers]
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Style
    colors = {"rich": "#2E86AB", "lazy": "#E94F37"}
    
    # Plot 1: Effective Rank vs Depth
    axes[0, 0].plot(layers, rich_ranks, 'o-', linewidth=2, markersize=8, 
                    color=colors["rich"], label="Rich Regime")
    axes[0, 0].plot(layers, lazy_ranks, 's--', linewidth=2, markersize=8, 
                    color=colors["lazy"], label="Lazy Regime")
    axes[0, 0].set_xlabel("Layers", fontsize=11)
    axes[0, 0].set_ylabel("Effective Rank", fontsize=11)
    axes[0, 0].set_title("Effective Rank vs Depth", fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy vs Depth
    axes[0, 1].plot(layers, rich_val_acc, 'o-', linewidth=2, markersize=8, 
                    color=colors["rich"], label="Rich Regime")
    axes[0, 1].plot(layers, lazy_val_acc, 's--', linewidth=2, markersize=8, 
                    color=colors["lazy"], label="Lazy Regime")
    axes[0, 1].set_xlabel("Layers", fontsize=11)
    axes[0, 1].set_ylabel("Validation Accuracy (%)", fontsize=11)
    axes[0, 1].set_title("Validation Accuracy vs Depth", fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: P⊥-pC vs Depth
    axes[1, 0].plot(layers, rich_p_perp, 'o-', linewidth=2, markersize=8, 
                    color=colors["rich"], label="Rich Regime")
    axes[1, 0].plot(layers, lazy_p_perp, 's--', linewidth=2, markersize=8, 
                    color=colors["lazy"], label="Lazy Regime")
    axes[1, 0].set_xlabel("Layers", fontsize=11)
    axes[1, 0].set_ylabel("P⊥-pC (%)", fontsize=11)
    axes[1, 0].set_title("P⊥-pC vs Depth (Lower = Stronger LD-SB)", fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: P-pC vs Depth
    axes[1, 1].plot(layers, rich_p_pc, 'o-', linewidth=2, markersize=8, 
                    color=colors["rich"], label="Rich Regime")
    axes[1, 1].plot(layers, lazy_p_pc, 's--', linewidth=2, markersize=8, 
                    color=colors["lazy"], label="Lazy Regime")
    axes[1, 1].set_xlabel("Layers", fontsize=11)
    axes[1, 1].set_ylabel("P-pC (%)", fontsize=11)
    axes[1, 1].set_title("P-pC vs Depth", fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / "plot_rich_vs_lazy.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path / 'plot_rich_vs_lazy.png'}")


def plot_rank_comparison(results, save_dir="results"):
    """Generate a focused plot comparing effective rank dynamics."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    rich = results["rich"]
    lazy = results["lazy"]
    
    if not rich or not lazy:
        return
    
    layers = sorted(set(rich.keys()) & set(lazy.keys()))
    
    rich_ranks = [rich[l]["final_effective_rank"] for l in layers]
    lazy_ranks = [lazy[l]["final_effective_rank"] for l in layers]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(layers))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rich_ranks, width, label='Rich Regime', color='#2E86AB', alpha=0.85)
    bars2 = ax.bar(x + width/2, lazy_ranks, width, label='Lazy Regime', color='#E94F37', alpha=0.85)
    
    ax.set_xlabel('Number of Layers', fontsize=12)
    ax.set_ylabel('Effective Rank', fontsize=12)
    ax.set_title('Effective Rank: Rich vs Lazy Regime', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path / "plot_rank_comparison_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path / 'plot_rank_comparison_bar.png'}")


def plot_training_curves(results, save_dir="results"):
    """Plot training curves (effective rank over time) for Rich regime."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    rich = results["rich"]
    if not rich:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(rich)))
    
    for i, num_layers in enumerate(sorted(rich.keys())):
        data = rich[num_layers]
        eff_rank_history = data["history"]["effective_rank"]
        steps = np.linspace(0, 20000, len(eff_rank_history))
        ax.plot(steps, eff_rank_history, label=f"{num_layers} layer(s)", 
                linewidth=1.5, color=colors[i])
    
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Effective Rank", fontsize=11)
    ax.set_title("Effective Rank During Training (Rich Regime)", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / "plot_rank_over_time.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path / 'plot_rank_over_time.png'}")


def print_summary_table(results):
    """Print a markdown summary table of all results."""
    print("\n## Rich vs Lazy Comparison Table\n")
    print("| Layers | Regime | Val Acc (%) | Eff. Rank | P⊥-pC (%) | P-pC (%) |")
    print("|--------|--------|-------------|-----------|-----------|----------|")
    
    layers = sorted(set(results["rich"].keys()) | set(results["lazy"].keys()))
    
    for num_layers in layers:
        for regime in ["rich", "lazy"]:
            if num_layers in results[regime]:
                data = results[regime][num_layers]
                val_acc = data["final_val_acc"]
                eff_rank = data["final_effective_rank"]
                p_perp_pc = data["ldsb_metrics"]["P_perp_pC"]
                p_pc = data["ldsb_metrics"]["P_pC"]
                
                print(f"| {num_layers:<6} | {regime:<6} | {val_acc:>11.2f} | {eff_rank:>9.2f} | {p_perp_pc:>9.1f} | {p_pc:>8.1f} |")


def generate_latex_table(results):
    """Generate LaTeX table for the paper."""
    print("\n%% LaTeX Table (copy into paper)\n")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Rich vs Lazy regime comparison across network depths on Waterbirds.}")
    print(r"\label{tab:rich_vs_lazy}")
    print(r"\begin{tabular}{cccccc}")
    print(r"\toprule")
    print(r"Layers & Regime & Val Acc (\%) & Eff. Rank & $P_{\perp}$-pC (\%) & $P$-pC (\%) \\")
    print(r"\midrule")
    
    layers = sorted(set(results["rich"].keys()) | set(results["lazy"].keys()))
    
    for num_layers in layers:
        for regime in ["rich", "lazy"]:
            if num_layers in results[regime]:
                data = results[regime][num_layers]
                val_acc = data["final_val_acc"]
                eff_rank = data["final_effective_rank"]
                p_perp_pc = data["ldsb_metrics"]["P_perp_pC"]
                p_pc = data["ldsb_metrics"]["P_pC"]
                
                regime_display = regime.capitalize()
                print(f"{num_layers} & {regime_display} & {val_acc:.2f} & {eff_rank:.2f} & {p_perp_pc:.1f} & {p_pc:.1f} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    # Load results
    results = load_results("outputs")
    
    total_results = len(results["rich"]) + len(results["lazy"])
    if total_results == 0:
        print("No results found in outputs/. Run experiments first:")
        print("  python main.py --regime rich --layers 1")
        print("  python main.py --regime lazy --layers 1")
        exit(1)
    
    print(f"Found Rich regime results: layers = {sorted(results['rich'].keys())}")
    print(f"Found Lazy regime results: layers = {sorted(results['lazy'].keys())}")
    
    # Generate plots
    plot_rich_vs_lazy_comparison(results)
    plot_rank_comparison(results)
    plot_training_curves(results)
    
    # Print summary
    print_summary_table(results)
    
    # Generate LaTeX table
    generate_latex_table(results)
