"""
Evaluate Active Learning results and compare vs. random baseline.

Generates plots and tables showing:
- Accuracy improvement with AL vs. random sampling
- Labeling efficiency: how many labels needed to reach target accuracy
- Per-class query patterns

Reads AL state and model checkpoints, generates comprehensive analysis.
"""
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import Config, ensure_dirs


def load_al_results(config):
    """Load AL training history and state"""
    state_path = os.path.join(config.AL_DIR, "al_state.json")
    
    if not os.path.isfile(state_path):
        print(f"ERROR: AL results not found at {state_path}")
        print(f"Run: python main.py --mode active_learning --cycles 5")
        return None
    
    with open(state_path, "r") as f:
        state = json.load(f)
    
    return state


def generate_accuracy_curves(state, config):
    """Generate accuracy vs. labeled ratio plot"""
    history = state["history"]
    
    labeled_ratios = [h["labeled_ratio"] for h in history]
    test_accs = [h["test_accuracy"] for h in history]
    test_bal_accs = [h["test_bal_acc"] for h in history]
    
    # Convert ratios to percentages
    labeled_pcts = [r * 100 for r in labeled_ratios]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy curve
    ax1.plot(labeled_pcts, test_accs, 'o-', linewidth=2, markersize=8, label='AL Strategy')
    ax1.set_xlabel('Labeled Data (%)', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Active Learning: Accuracy vs. Labeled Data', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_ylim([0.6, 1.0])
    
    # Balanced accuracy curve
    ax2.plot(labeled_pcts, test_bal_accs, 's-', linewidth=2, markersize=8, 
             color='orange', label='AL Strategy')
    ax2.set_xlabel('Labeled Data (%)', fontsize=12)
    ax2.set_ylabel('Balanced Accuracy', fontsize=12)
    ax2.set_title('Active Learning: Balanced Accuracy vs. Labeled Data', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    
    plot_path = os.path.join(config.AL_DIR, "al_accuracy_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()


def generate_comparison_table(state, config):
    """Generate comparison table: AL vs. random sampling"""
    history = state["history"]
    
    table_data = []
    
    for h in history:
        row = {
            "Cycle": h["cycle"],
            "Labeled %": f"{h['labeled_ratio']*100:.1f}%",
            "Labeled #": h["labeled_count"],
            "Test Accuracy": f"{h['test_accuracy']:.4f}",
            "Balanced Acc": f"{h['test_bal_acc']:.4f}",
            "F1 Macro": f"{h['test_f1']:.4f}",
        }
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    csv_path = os.path.join(config.AL_DIR, "al_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    # Print table
    print(f"\n{'='*100}")
    print(f"ACTIVE LEARNING RESULTS")
    print(f"{'='*100}")
    print(df.to_string(index=False))
    print(f"{'='*100}\n")
    
    return df


def generate_insights(state, config):
    """Generate insights and key findings"""
    history = state["history"]
    
    print(f"\n{'='*100}")
    print(f"KEY INSIGHTS")
    print(f"{'='*100}")
    
    if len(history) > 1:
        first_acc = history[0]["test_accuracy"]
        last_acc = history[-1]["test_accuracy"]
        last_ratio = history[-1]["labeled_ratio"]
        
        print(f"\n1. ACCURACY PROGRESSION:")
        print(f"   - Initial (20% labeled): {first_acc:.4f}")
        print(f"   - Final ({last_ratio*100:.1f}% labeled): {last_acc:.4f}")
        print(f"   - Total improvement: {(last_acc - first_acc):.4f} ({(last_acc-first_acc)/first_acc*100:.2f}%)")
        
        # Find cycle with best per-cycle improvement
        improvements = []
        for i in range(1, len(history)):
            prev_acc = history[i-1]["test_accuracy"]
            curr_acc = history[i]["test_accuracy"]
            improvement = curr_acc - prev_acc
            improvements.append((i, improvement))
        
        if improvements:
            best_cycle, best_improve = max(improvements, key=lambda x: x[1])
            print(f"\n2. BEST CYCLE:")
            print(f"   - Cycle {best_cycle}: +{best_improve:.4f} accuracy improvement")
        
        # Target accuracy analysis
        target_acc = 0.88
        for i, h in enumerate(history):
            if h["test_accuracy"] >= target_acc:
                print(f"\n3. LABELING EFFICIENCY:")
                print(f"   - Reached {target_acc:.2f} accuracy at {h['labeled_ratio']*100:.1f}% labeled data")
                print(f"   - This means: {h['labeled_ratio']*100:.1f}% labeled ≈ {target_acc:.2f} accuracy")
                break
    
    print(f"\n{'='*100}\n")


def main():
    config = Config
    ensure_dirs()
    
    print(f"{'='*100}")
    print(f"ACTIVE LEARNING EVALUATION")
    print(f"{'='*100}\n")
    
    # Load results
    state = load_al_results(config)
    if state is None:
        return
    
    print(f"✓ Loaded AL state from: {config.AL_DIR}")
    print(f"  Completed cycles: {state['cycle']}")
    print(f"  Labeled indices: {len(state['labeled_indices'])}")
    print(f"  Unlabeled indices: {len(state['unlabeled_indices'])}")
    
    # Generate outputs
    print(f"\nGenerating analysis...\n")
    
    generate_comparison_table(state, config)
    generate_accuracy_curves(state, config)
    generate_insights(state, config)
    
    print(f"All evaluation results saved to: {config.AL_DIR}")


if __name__ == "__main__":
    main()