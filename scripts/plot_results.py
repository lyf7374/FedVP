"""Render the manuscript's N=1,2,4,8 candidate-budget results from saved JSON."""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "candidate_scaling.json")
    parser.add_argument("--output", type=Path, default=ROOT / "assets" / "candidate_scaling.png")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(args.input.read_text(encoding="utf-8"))
    budgets = data["candidate_budgets"]
    styles = {
        "FedVP (Uniform)": {"color": "#d62728", "marker": "o", "linewidth": 2.2},
        "FedVP (Heterogeneous)": {"color": "#9467bd", "marker": "D", "linestyle": "--", "linewidth": 2.2},
        "Self-Consistency": {"color": "#2ca02c", "marker": "s"},
        "Pretrained Verifier": {"color": "#1f77b4", "marker": "^"},
        "Oracle": {"color": "#7f7f7f", "marker": "o", "markerfacecolor": "white", "linestyle": "--"},
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3), sharey=True)
    for ax, (target, series) in zip(axes, data["series"].items()):
        for method, values in series.items():
            ax.plot(budgets, values, label=method, markersize=5, **styles[method])
        ax.set_title(target, fontsize=11)
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets, [str(n) for n in budgets])
        ax.set_xlabel("Candidates per question (N)")
        ax.set_ylim(52, 86)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Five-dataset mean accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
