"""
Plot LD-SB experiment results from JSON output files.
Generates figures for effective rank, P⊥-pC, P-pC, and validation accuracy vs depth.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def load_results(output_dir="outputs"):
    """Load all results_layer*.json files from the output directory."""
    results = {}
    output_path = Path(output_dir)
    
    for json_file in sorted(output_path.glob("results_layer*.json")):
        with open(json_file) as f:
            data = json.load(f)
            num_layers = data["config"]["num_layers"]
            results[num_layers] = data
    
    return results


def plot_all_metrics(results, save_dir="results"):
    """Generate a 2x2 figure with all key metrics."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Sort by number of layers
    layers = sorted(results.keys())
    
    # Extract metrics
    eff_ranks = [results[l]["final_effective_rank"] for l in layers]
    p_perp_pc = [results[l]["ldsb_metrics"]["P_perp_pC"] for l in layers]
    p_pc = [results[l]["ldsb_metrics"]["P_pC"] for l in layers]
    val_accs = [results[l]["final_val_acc"] for l in layers]
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Effective Rank vs Depth
    axes[0, 0].plot(layers, eff_ranks, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel("Layers")
    axes[0, 0].set_ylabel("Eff. Rank")
    axes[0, 0].set_title("Effective Rank vs Depth")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: P⊥-pC vs Depth
    axes[0, 1].plot(layers, p_perp_pc, 'o-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel("Layers")
    axes[0, 1].set_ylabel("P⊥-pC (%)")
    axes[0, 1].set_title("P⊥-pC vs Depth")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: P-pC vs Depth
    axes[1, 0].plot(layers, p_pc, 'o-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel("Layers")
    axes[1, 0].set_ylabel("P-pC (%)")
    axes[1, 0].set_title("P-pC vs Depth")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Validation Accuracy vs Depth
    axes[1, 1].plot(layers, val_accs, 'o-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel("Layers")
    axes[1, 1].set_ylabel("Val Acc (%)")
    axes[1, 1].set_title("Validation Accuracy vs Depth")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / "plot_all_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path / 'plot_all_metrics.png'}")


def plot_individual_metrics(results, save_dir="results"):
    """Generate individual plots for each metric."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    layers = sorted(results.keys())
    
    metrics = {
        "effective_rank": {
            "data": [results[l]["final_effective_rank"] for l in layers],
            "ylabel": "Effective Rank",
            "title": "Effective Rank vs Depth"
        },
        "p_perp_pc": {
            "data": [results[l]["ldsb_metrics"]["P_perp_pC"] for l in layers],
            "ylabel": "P⊥-pC (%)",
            "title": "P⊥-pC vs Depth"
        },
        "p_pc": {
            "data": [results[l]["ldsb_metrics"]["P_pC"] for l in layers],
            "ylabel": "P-pC (%)",
            "title": "P-pC vs Depth"
        },
        "val_acc": {
            "data": [results[l]["final_val_acc"] for l in layers],
            "ylabel": "Validation Accuracy (%)",
            "title": "Validation Accuracy vs Depth"
        }
    }
    
    for name, info in metrics.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(layers, info["data"], 'o-', linewidth=2, markersize=8)
        ax.set_xlabel("Layers")
        ax.set_ylabel(info["ylabel"])
        ax.set_title(info["title"])
        ax.grid(True, alpha=0.3)
        
        filename = save_path / f"plot_{name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")


def plot_training_curves(results, save_dir="results"):
    """Plot training curves (effective rank over time) for all experiments."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for num_layers in sorted(results.keys()):
        data = results[num_layers]
        eff_rank_history = data["history"]["effective_rank"]
        epochs = range(1, len(eff_rank_history) + 1)
        ax.plot(epochs, eff_rank_history, label=f"{num_layers} layer(s)", linewidth=1.5)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Effective Rank")
    ax.set_title("Effective Rank During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    filename = save_path / "plot_rank_over_time.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def print_summary_table(results):
    """Print a markdown summary table of all results."""
    print("\n## Summary Table\n")
    print("| Layers | Learning Rate | Final Val Acc | Final Eff. Rank | P⊥-pC (↓) | P-pC (↑) | rank(P) |")
    print("|--------|---------------|---------------|-----------------|-----------|----------|---------|")
    
    for num_layers in sorted(results.keys()):
        data = results[num_layers]
        lr = data["config"]["learning_rate"]
        val_acc = data["final_val_acc"]
        eff_rank = data["final_effective_rank"]
        p_perp_pc = data["ldsb_metrics"]["P_perp_pC"]
        p_pc = data["ldsb_metrics"]["P_pC"]
        rank_p = data["ldsb_metrics"]["rank_P"]
        
        print(f"| {num_layers:<6} | {lr:<13} | {val_acc:.2f}%{' '*(6-len(f'{val_acc:.2f}'))} | {eff_rank:.2f}{' '*(14-len(f'{eff_rank:.2f}'))}| {p_perp_pc:.1f}%{' '*(7-len(f'{p_perp_pc:.1f}'))}| {p_pc:.1f}%{' '*(6-len(f'{p_pc:.1f}'))}| {rank_p:<7} |")


if __name__ == "__main__":
    # Load results
    results = load_results("outputs")
    
    if not results:
        print("No results found in outputs/. Run experiments first:")
        print("  python main.py --layers 1 --lr 1.0")
        print("  python main.py --layers 5 --lr 0.2")
        print("  python main.py --layers 10 --lr 0.1")
        print("  python main.py --layers 20 --lr 0.05")
        print("  python main.py --layers 50 --lr 0.01")
        exit(1)
    
    print(f"Found results for {len(results)} experiment(s): layers = {sorted(results.keys())}")
    
    # Generate plots
    plot_all_metrics(results)
    plot_individual_metrics(results)
    plot_training_curves(results)
    
    # Print summary
    print_summary_table(results)

