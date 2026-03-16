# Among Us LLM Experiments

Code of: **"Probing Generative Models Through Adversarial Social Games: Deception, Memory, and Reasoning in Among Us"**

## Project Structure

```
among_us_llm/
├── game/
│   ├── environment.py    # 14-room map, GameState, Player dataclasses
│   └── mechanics.py      # Action execution, meeting phase, vote resolution
├── agents/
│   ├── llm_agent.py      # LLMAgent (GPT + function calling) & RandomAgent
│   ├── memory.py         # Rolling memory with LLM compression
│   └── personalities.py  # 10 personality archetypes (5 per role)
├── experiments/
│   └── runner.py         # Factorial design (A1-A4) + ablation studies
├── analysis/
│   ├── evaluator.py      # Cognitive evaluation + speech classification
│   └── visualize.py      # Paper-style figures (matplotlib)
├── main.py               # CLI entry point
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
```

## Usage

### Run a single demo game

```bash
# All LLM agents (default)
python main.py --mode single --config all_llm --verbose

# LLM Crewmates vs Random Impostor
python main.py --mode single --config llm_crew

# All random (baseline)
python main.py --mode single --config all_random
```

### Replicate main experiments (paper Table 2)

```bash
# Full replication: A1-A4 configurations
python main.py --mode main --model gpt-4o-mini

# Quick test (3 games per config)
python main.py --mode main --quick
```

### Ablation studies

```bash
# Memory size + planning prompt ablations
python main.py --mode ablations --quick
```

### Generate figures

```bash
# Requires results in ./results/ directory
python main.py --mode figures
```

### Run everything

```bash
python main.py --mode all --quick
```

## Experimental Configurations

| Config | Crewmate Agent | Impostor Agent | What It Probes |
|--------|----------------|----------------|----------------|
| A1: All-Random | Random | Random | Baseline (no model) |
| A2: All-LLM | GPT | GPT | Full model capabilities |
| A3: LLM-Crew | GPT | Random | Deception detection |
| A4: LLM-Impostor | Random | GPT | Deception generation |

## Key Findings (from paper)

1. **Deception asymmetry**: 93% Crewmate win vs Random Impostors, only 36% vs LLM Impostors
2. **Memory is necessary**: Win rate scales 0% → 20% → 36% with Nmax = 0, 5, 10
3. **Planning is not the bottleneck**: ReAct vs minimal prompt: 36% vs 33% (no significant difference)

## Model Notes

The paper uses **GPT-5-mini**; this code defaults to `gpt-4o-mini` (the closest available equivalent). Change with `--model gpt-4o` for stronger results.

## Cost Estimate

~$0.10–0.30 per game with gpt-4o-mini. Full replication (N=76 games) ≈ $8–23.

## Output

Results are saved as JSON files in `results/`:

- `results/main/A1_all_random.json`
- `results/main/A2_all_llm.json`
- `results/ablations/mem_0.json` ... etc.
- `results/figures/*.png` (generated figures)
