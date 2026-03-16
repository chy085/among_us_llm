"""
Experiment runner: replicates the paper's factorial design and ablation studies.
Runs N games per configuration and collects statistics.
"""

import os
import json
import time
from typing import List, Dict, Optional
from pathlib import Path
from game.runner import run_single_game


PLAYER_NAMES = ["Alice", "Bob", "Charlie", "Diana", "Evan"]

# Paper's experimental configurations
MAIN_CONFIGS = {
    "A1_all_random":   {"config": "all_random",  "n_games": 10, "memory_size": 10, "use_react": True},
    "A2_all_llm":      {"config": "all_llm",      "n_games": 14, "memory_size": 10, "use_react": True},
    "A3_llm_crew":     {"config": "llm_crew",     "n_games": 14, "memory_size": 10, "use_react": True},
    "A4_llm_imp":      {"config": "llm_imp",      "n_games": 15, "memory_size": 10, "use_react": True},
}

# Ablation: memory size (all_llm config)
MEMORY_ABLATIONS = {
    "mem_0":  {"config": "all_llm", "n_games": 5,  "memory_size": 0,  "use_react": True},
    "mem_5":  {"config": "all_llm", "n_games": 5,  "memory_size": 5,  "use_react": True},
    "mem_10": {"config": "all_llm", "n_games": 14, "memory_size": 10, "use_react": True},
    "mem_20": {"config": "all_llm", "n_games": 7,  "memory_size": 20, "use_react": True},
}

# Ablation: planning prompt (all_llm config)
PLANNING_ABLATIONS = {
    "with_react":    {"config": "all_llm", "n_games": 14, "memory_size": 10, "use_react": True},
    "no_planning":   {"config": "all_llm", "n_games": 6,  "memory_size": 10, "use_react": False},
}


def run_experiment(
    exp_name: str,
    config: str,
    n_games: int,
    memory_size: int = 10,
    use_react: bool = True,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    output_dir: str = "results",
    verbose: bool = False,
) -> Dict:
    """Run N games for a configuration and return aggregated statistics."""
    print(f"\n{'#'*60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"  Config: {config} | N={n_games} | Memory={memory_size} | ReAct={use_react}")
    print(f"{'#'*60}")

    Path(output_dir).mkdir(exist_ok=True)

    results = []
    for i in range(n_games):
        print(f"\n[Game {i+1}/{n_games}]")
        try:
            result = run_single_game(
                config=config,
                player_names=PLAYER_NAMES,
                memory_size=memory_size,
                use_react=use_react,
                model=model,
                api_key=api_key,
                verbose=verbose,
            )
            results.append(result)
            print(f"  Winner: {result['winner']} | Steps: {result['timesteps']} | Kills: {result['kills']}")
        except Exception as e:
            print(f"  Game {i+1} failed: {e}")
            continue

        # Brief pause to avoid rate limits
        time.sleep(0.5)

    # Aggregate
    stats = aggregate_results(exp_name, config, results)

    # Save raw results
    out_path = Path(output_dir) / f"{exp_name}.json"
    with open(out_path, "w") as f:
        json.dump({"experiment": exp_name, "config": config, "stats": stats, "games": results}, f, indent=2)

    print(f"\n  Results saved to {out_path}")
    print(f"  Summary: Crew {stats['crew_win_rate']:.1%} | Imp {stats['imp_win_rate']:.1%} | "
          f"Avg Steps {stats['avg_steps']:.1f} | Avg Kills {stats['avg_kills']:.1f}")

    return stats


def aggregate_results(exp_name: str, config: str, results: List[Dict]) -> Dict:
    """Compute win rates and confidence intervals."""
    import math

    n = len(results)
    if n == 0:
        return {"n": 0, "crew_win_rate": 0, "imp_win_rate": 0}

    crew_wins = sum(1 for r in results if r.get("winner") == "Crewmate")
    imp_wins = sum(1 for r in results if r.get("winner") == "Impostor")

    crew_rate = crew_wins / n
    imp_rate = imp_wins / n

    # Wilson 95% CI
    z = 1.96
    def wilson_ci(p, n):
        if n == 0:
            return 0, 0
        center = (p + z**2 / (2*n)) / (1 + z**2 / n)
        margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / (1 + z**2/n)
        return max(0, center - margin), min(1, center + margin)

    crew_lo, crew_hi = wilson_ci(crew_rate, n)
    imp_lo, imp_hi = wilson_ci(imp_rate, n)

    avg_steps = sum(r.get("timesteps", 0) for r in results) / n
    avg_kills = sum(r.get("kills", 0) for r in results) / n
    avg_meetings = sum(r.get("total_meetings", 0) for r in results) / n

    return {
        "experiment": exp_name,
        "config": config,
        "n": n,
        "crew_wins": crew_wins,
        "imp_wins": imp_wins,
        "crew_win_rate": crew_rate,
        "imp_win_rate": imp_rate,
        "crew_ci_95": f"±{(crew_hi - crew_lo) / 2:.2%}",
        "imp_ci_95": f"±{(imp_hi - imp_lo) / 2:.2%}",
        "crew_ci_low": crew_lo,
        "crew_ci_high": crew_hi,
        "avg_steps": avg_steps,
        "avg_kills": avg_kills,
        "avg_meetings": avg_meetings,
    }


def run_all_main_experiments(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    output_dir: str = "results/main",
    verbose: bool = False,
    quick_mode: bool = False,  # reduce n_games for testing
):
    """Run all 4 main configurations."""
    all_stats = {}
    configs = MAIN_CONFIGS.copy()

    if quick_mode:
        for k in configs:
            configs[k] = {**configs[k], "n_games": 3}

    for exp_name, params in configs.items():
        stats = run_experiment(
            exp_name=exp_name,
            model=model,
            api_key=api_key,
            output_dir=output_dir,
            verbose=verbose,
            **params,
        )
        all_stats[exp_name] = stats

    print_results_table(all_stats)
    return all_stats


def run_memory_ablations(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    output_dir: str = "results/ablations",
    verbose: bool = False,
    quick_mode: bool = False,
):
    """Run memory size ablation studies."""
    all_stats = {}
    ablations = MEMORY_ABLATIONS.copy()

    if quick_mode:
        for k in ablations:
            ablations[k] = {**ablations[k], "n_games": 3}

    for exp_name, params in ablations.items():
        stats = run_experiment(
            exp_name=exp_name,
            model=model,
            api_key=api_key,
            output_dir=output_dir,
            verbose=verbose,
            **params,
        )
        all_stats[exp_name] = stats

    print("\n=== Memory Ablation Results ===")
    print(f"{'Memory':>10} | {'N':>4} | {'Crew%':>8} | {'Imp%':>8} | {'Steps':>7}")
    print("-" * 50)
    for name, s in all_stats.items():
        mem = name.replace("mem_", "Nmax=")
        print(f"{mem:>10} | {s['n']:>4} | {s['crew_win_rate']:>8.1%} | {s['imp_win_rate']:>8.1%} | {s['avg_steps']:>7.1f}")

    return all_stats


def run_planning_ablations(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    output_dir: str = "results/ablations",
    verbose: bool = False,
    quick_mode: bool = False,
):
    """Run planning prompt ablation studies."""
    all_stats = {}
    ablations = PLANNING_ABLATIONS.copy()

    if quick_mode:
        for k in ablations:
            ablations[k] = {**ablations[k], "n_games": 3}

    for exp_name, params in ablations.items():
        stats = run_experiment(
            exp_name=exp_name,
            model=model,
            api_key=api_key,
            output_dir=output_dir,
            verbose=verbose,
            **params,
        )
        all_stats[exp_name] = stats

    print("\n=== Planning Ablation Results ===")
    print(f"{'Config':>15} | {'N':>4} | {'Crew%':>8} | {'Imp%':>8}")
    print("-" * 45)
    for name, s in all_stats.items():
        print(f"{name:>15} | {s['n']:>4} | {s['crew_win_rate']:>8.1%} | {s['imp_win_rate']:>8.1%}")

    return all_stats


def print_results_table(all_stats: Dict):
    """Print results in paper format."""
    print("\n" + "="*70)
    print("TABLE 2: Win rates across experimental configurations")
    print("="*70)
    print(f"{'Config':<20} | {'N':>4} | {'Crew%':>8} | {'Imp%':>8} | {'Steps':>7} | {'Kills':>6}")
    print("-" * 70)
    for name, s in all_stats.items():
        label = name.replace("_", " ").title()
        ci = s.get('crew_ci_95', '')
        print(f"{label:<20} | {s['n']:>4} | {s['crew_win_rate']:>6.0%}{ci:>4} | "
              f"{s['imp_win_rate']:>6.0%} | {s['avg_steps']:>7.1f} | {s['avg_kills']:>6.1f}")
    print("="*70)
