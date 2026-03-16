"""
Main entry point for Among Us LLM experiments.

Usage:
  python main.py --mode single --config all_llm
  python main.py --mode main --quick
  python main.py --mode ablations
  python main.py --mode figures
  python main.py --mode all --quick
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Among Us LLM Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  single      Run a single game (demo)
  main        Run all 4 main configurations (A1-A4)
  ablations   Run memory and planning ablations
  figures     Generate figures from saved results
  all         Run everything

Examples:
  python main.py --mode single --config all_llm --verbose
  python main.py --mode main --quick --model gpt-4o-mini
  python main.py --mode all --quick
  python main.py --mode figures
        """
    )

    parser.add_argument(
        "--mode", choices=["single", "main", "ablations", "figures", "all"],
        default="single", help="What to run"
    )
    parser.add_argument(
        "--config",
        choices=["all_random", "all_llm", "llm_crew", "llm_imp"],
        default="all_llm", help="Config for single-game mode"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--api-key", default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--memory", type=int, default=10,
        help="Memory size for single-game mode (default: 10)"
    )
    parser.add_argument(
        "--no-react", action="store_true",
        help="Disable ReAct-style planning"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: reduce games per config to 2-3"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full game logs"
    )
    parser.add_argument(
        "--output", default="results",
        help="Output directory for results (default: results)"
    )

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if args.mode != "figures" and not api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY or pass --api-key.")
        sys.exit(1)

    if args.mode == "single":
        run_single(args, api_key)
    elif args.mode == "main":
        run_main_experiments(args, api_key)
    elif args.mode == "ablations":
        run_ablations(args, api_key)
    elif args.mode == "figures":
        run_figures(args)
    elif args.mode == "all":
        run_all(args, api_key)


def run_single(args, api_key):
    print(f"\nRunning single game: {args.config}")
    from game.runner import run_single_game

    result = run_single_game(
        config=args.config,
        memory_size=args.memory,
        use_react=not args.no_react,
        model=args.model,
        api_key=api_key,
        verbose=True,
    )
    print(f"\n{'='*60}")
    print(f"RESULT: {result['winner']}s WIN!")
    print(f"Steps: {result['timesteps']} | Kills: {result['kills']} | Meetings: {result['total_meetings']}")


def run_main_experiments(args, api_key):
    from experiments.runner import run_all_main_experiments

    run_all_main_experiments(
        model=args.model,
        api_key=api_key,
        output_dir=f"{args.output}/main",
        verbose=args.verbose,
        quick_mode=args.quick,
    )


def run_ablations(args, api_key):
    from experiments.runner import run_memory_ablations, run_planning_ablations

    run_memory_ablations(
        model=args.model,
        api_key=api_key,
        output_dir=f"{args.output}/ablations",
        verbose=args.verbose,
        quick_mode=args.quick,
    )
    run_planning_ablations(
        model=args.model,
        api_key=api_key,
        output_dir=f"{args.output}/ablations",
        verbose=args.verbose,
        quick_mode=args.quick,
    )


def run_figures(args):
    from analysis.visualize import generate_all_figures

    generate_all_figures(
        results_dir=args.output,
        output_dir=f"{args.output}/figures",
    )


def run_all(args, api_key):
    run_main_experiments(args, api_key)
    run_ablations(args, api_key)
    run_figures(args)


if __name__ == "__main__":
    main()
