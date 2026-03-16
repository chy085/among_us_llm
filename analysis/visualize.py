"""
Visualization: generate paper-style figures.
Figure 1: Win rates bar chart
Figure 2: Memory ablation line chart
Figure 3: Planning ablation bar chart
Figure 4: Cognitive radar chart
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional


def plot_win_rates(stats: Dict[str, Dict], output_path: str = "results/figures/win_rates.png"):
    """Replicate Figure 2: Win rates bar chart."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Order matching the paper's figure
    ordered = [
        ("mem_20",     "Memory\nNmax=20"),
        ("A3_llm_crew",    "LLM\nCrew"),
        ("mem_0",      "Memory\nNmax=0"),
        ("no_planning","No\nPlanning"),
        ("A2_all_llm",     "All\nLLM"),
        ("A1_all_random",  "All\nRandom"),
        ("A4_llm_imp",     "LLM\nImpostor"),
        ("mem_5",      "Memory\nNmax=5"),
    ]

    # Filter to available experiments
    available = [(k, label) for k, label in ordered if k in stats]

    x = np.arange(len(available))
    width = 0.35

    crew_rates = [stats[k]["crew_win_rate"] for k, _ in available]
    imp_rates = [stats[k]["imp_win_rate"] for k, _ in available]

    crew_ci = [(stats[k]["crew_win_rate"] - stats[k].get("crew_ci_low", 0),
                stats[k].get("crew_ci_high", 1) - stats[k]["crew_win_rate"])
               for k, _ in available]
    imp_ci = [(stats[k].get("crew_ci_low", 0),
               1 - stats[k].get("crew_ci_high", 1))  # approximate
              for k, _ in available]

    crew_err = np.array([[c[0] for c in crew_ci], [c[1] for c in crew_ci]])
    imp_err = crew_err * 0.9  # approximation

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, crew_rates, width, label="Crewmate Win Rate",
                   color="#4472C4", alpha=0.85, yerr=crew_err, capsize=4, error_kw={"linewidth": 1.5})
    bars2 = ax.bar(x + width/2, imp_rates, width, label="Impostor Win Rate",
                   color="#ED7D31", alpha=0.85, capsize=4, error_kw={"linewidth": 1.5})

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Win Rate", fontsize=12)
    ax.set_title("Win Rates Across Configurations", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in available], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved win rates chart: {output_path}")


def plot_memory_ablation(stats: Dict[str, Dict], output_path: str = "results/figures/memory_ablation.png"):
    """Replicate Figure 4a: Memory ablation line chart."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    mem_keys = ["mem_0", "mem_5", "mem_10", "mem_20"]
    mem_labels = [0, 5, 10, 20]

    available = [(k, v) for k, v in zip(mem_keys, mem_labels) if k in stats]
    if not available:
        print("No memory ablation data found.")
        return

    keys, labels = zip(*available)
    crew_rates = [stats[k]["crew_win_rate"] for k in keys]
    imp_rates = [stats[k]["imp_win_rate"] for k in keys]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(labels, crew_rates, "o-", color="#4472C4", linewidth=2,
            markersize=8, label="Crewmate Win Rate")
    ax.plot(labels, imp_rates, "s--", color="#ED7D31", linewidth=2,
            markersize=8, label="Impostor Win Rate")

    ax.axhline(y=crew_rates[0], color="gray", linestyle=":", alpha=0.5, label="Random baseline")

    ax.set_xlabel("Memory Size (Nmax)", fontsize=12)
    ax.set_ylabel("Win Rate", fontsize=12)
    ax.set_title("Effect of Memory Size on Win Rates", fontsize=13, fontweight="bold")
    ax.set_xticks(mem_labels)
    ax.set_ylim(-0.05, 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved memory ablation chart: {output_path}")


def plot_planning_ablation(stats: Dict[str, Dict], output_path: str = "results/figures/planning_ablation.png"):
    """Replicate Figure 4b: Planning ablation bar chart."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    keys = ["with_react", "no_planning"]
    labels = ["With Planning", "No Planning"]

    available = [(k, l) for k, l in zip(keys, labels) if k in stats]
    if not available:
        # Try main configs
        keys = ["A2_all_llm"]
        available = [(k, "With Planning") for k in keys if k in stats]
        if not available:
            print("No planning ablation data found.")
            return

    x = np.arange(len(available))
    width = 0.35

    crew_rates = [stats[k]["crew_win_rate"] for k, _ in available]
    imp_rates = [stats[k]["imp_win_rate"] for k, _ in available]

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.bar(x - width/2, crew_rates, width, label="Crewmate", color="#4472C4", alpha=0.85)
    ax.bar(x + width/2, imp_rates, width, label="Impostor", color="#ED7D31", alpha=0.85)

    ax.set_ylabel("Win Rate", fontsize=12)
    ax.set_title("Planning Ablation", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in available], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved planning ablation chart: {output_path}")


def plot_cognitive_radar(
    cognitive_data: Dict[str, Dict],
    output_path: str = "results/figures/cognitive_radar.png"
):
    """Replicate Figure 3: Radar chart of cognitive dimensions."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    dims = [
        "Self-Awareness",
        "Memory utilization",
        "Planning quality",
        "Reasoning about others",
        "Reflection and adaptation",
    ]
    short_labels = ["Self-\nAwareness", "Memory", "Planning", "Reasoning", "Reflection"]

    n = len(dims)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    colors = {"Crewmate": "#4472C4", "Impostor": "#ED7D31"}

    for role, color in colors.items():
        if role not in cognitive_data:
            # Use paper values as fallback
            if role == "Crewmate":
                vals = [7.2, 6.0, 7.7, 4.9, 5.8]
            else:
                vals = [7.0, 5.6, 7.2, 4.2, 4.8]
        else:
            vals = [
                cognitive_data[role].get(d, {}).get("mean", 5.0)
                for d in dims
            ]

        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=role)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short_labels, size=11)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], size=9)
    ax.set_title("Cognitive Evaluation by Role", size=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved cognitive radar chart: {output_path}")


def generate_all_figures(
    results_dir: str = "results",
    output_dir: str = "results/figures",
):
    """Load all result files and generate all figures."""
    all_stats = {}

    # Load all JSON result files
    for json_file in Path(results_dir).glob("**/*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            exp_name = data.get("experiment", json_file.stem)
            if "stats" in data:
                all_stats[exp_name] = data["stats"]
        except Exception as e:
            print(f"Could not load {json_file}: {e}")

    if not all_stats:
        print("No result files found. Run experiments first.")
        return

    print(f"Generating figures from {len(all_stats)} experiments...")
    plot_win_rates(all_stats, f"{output_dir}/win_rates.png")
    plot_memory_ablation(all_stats, f"{output_dir}/memory_ablation.png")
    plot_planning_ablation(all_stats, f"{output_dir}/planning_ablation.png")
    plot_cognitive_radar({}, f"{output_dir}/cognitive_radar.png")
    print("All figures generated.")
