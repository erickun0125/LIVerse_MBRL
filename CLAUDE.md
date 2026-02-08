# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LIVerse (Language & Image integrated uniVerse) is a PyTorch-based research project implementing a world foundation model that unifies vision-language semantics and forward dynamics for model-based reinforcement learning. It introduces a PlaNet-style MPC agent using VLM-based rewards (via LIV/CLIP) for zero-shot policy execution on robotic manipulation tasks.

Built upon: DreamerV3 (world model), dreamerv3-torch (PyTorch port), LIV (vision-language model), PlaNet (MPC).

## Repository Structure

```
├── code_workspace/          # All source code, configs, and environments
│   ├── dreamer.py           # World model training entry point
│   ├── planet_basic.py      # Single-goal MPC planner
│   ├── planet_stepwise.py   # Multi-subtask MPC planner
│   ├── models.py            # World model architecture
│   ├── networks.py          # Neural network building blocks
│   ├── configs.yaml         # Training configuration
│   ├── configs_planet.yaml  # MPC planning configuration
│   ├── liv/                 # LIV vision-language model
│   ├── Metaworld/           # Meta-World benchmark (local copy)
│   └── envs/                # Environment wrappers
└── paper_workspace/         # LaTeX thesis and figures
    ├── main.tex             # Thesis source (Korean, uses kotex)
    └── *.png                # Figures for the paper
```

## Commands

All commands run from `code_workspace/`.

### Installation
```bash
pip install -r code_workspace/requirements.txt
```

### Training World Model
```bash
python code_workspace/dreamer.py --configs metaworld --task ML1_button-press-v2 --logdir ./logdir/button_press
python code_workspace/dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/walker_walk
```

### MPC Planning (Zero-Shot)
```bash
# Basic single-goal
python code_workspace/planet_basic.py --configs metaworld --task ML1_button-press-v2 \
    --logdir ./logdir/button_press \
    --planning_horizon 12 --candidates 1000 --goal_image "success_frame_button" --reward_form "similarity"

# Step-wise multi-subtask (delta-score rewards)
python code_workspace/planet_stepwise.py --configs metaworld --task ML1_button-press-v2 \
    --logdir ./logdir/button_press \
    --step_wise True --subtask_dir "./subtasks" --reward_form "difference"
```

### Evaluation
```bash
python code_workspace/test.py --logdir ./logdir/button_press --task ML1_button-press-v2 --configs metaworld
python code_workspace/simulate_episode.py
```

### LaTeX Paper Build
```bash
cd paper_workspace && latexmk -pdf main.tex
```

### Monitoring
```bash
tensorboard --logdir ./logdir
```

## Code Architecture

### Core Components (`code_workspace/`)

- **`dreamer.py`** — Main Dreamer agent: world model training loop, environment interaction, replay buffer, VLM reward integration.
- **`models.py`** — WorldModel class: encoder, RSSM dynamics, decoder/reward/continuation heads. ImagBehavior: actor-critic for imagination-based policy learning.
- **`networks.py`** — Neural network building blocks: RSSM (GRU-based recurrent state-space model), MultiEncoder/MultiDecoder, MLP, CNN, discrete/continuous latent distributions.
- **`planet_basic.py`** — Single-goal MPC planner using Cross-Entropy Method (CEM). Optimizes action sequences via cosine similarity reward with goal embedding.
- **`planet_stepwise.py`** — Multi-subtask MPC with delta-score reward function (incremental similarity improvement). Includes adaptive goal switching via plateau detection.

### VLM Integration

- **`liv/`** — LIV vision-language model (CLIP/ResNet50 backbone). Encodes images and text into shared embedding space. Auto-downloads from HuggingFace Hub on first use (`~/.liv/resnet50/`). Configured via Hydra (`liv/cfgs/`).
- **`metaworld_wrapper.py`** — Wraps Meta-World ML1 tasks for Dreamer compatibility. Returns dict observations: `image`, `raw_obs`, `discount`, `is_first`, `is_terminal`, `target_image_embedding`.

### Environment Support

- **`envs/`** — Wrappers for DMC, Atari, Crafter, Minecraft, MemoryMaze. Common wrappers in `envs/wrappers.py` (TimeLimit, NormalizeActions, OneHotAction).
- **`Metaworld/`** — Local copy of Meta-World benchmark (robotic manipulation tasks).
- **`parallel.py`** — Multi-process environment execution via `ProcessPipeWorker`.

### Configuration System

Two YAML config files with hierarchical override:
1. **`configs.yaml`** — Training configs. Base `defaults` section merged with suite-specific presets (`dmc_vision`, `dmc_proprio`, `metaworld`, `atari100k`, `crafter`, `minecraft`, `memorymaze`, `debug`). CLI args override all.
2. **`configs_planet.yaml`** — MPC planning configs with Meta-World specific parameters.

Config resolution order: `defaults` → suite preset → CLI overrides.

### Key Design Patterns

- **RSSM latent state**: Split into deterministic (`deter`) and stochastic (`stoch`) components. Discrete latent uses 32x32 categorical; continuous uses Gaussian.
- **Reward computation**: Cosine similarity between predicted latent embedding and LIV-encoded goal (image or text). Delta-score variant: `Δsim = sim_t - sim_{t-1}`.
- **CEM optimization**: Samples N candidate action sequences, evaluates via world model rollout, refits distribution to top-K performers. Iterates for refinement.
- **Symmetric log transform**: `symlog(x)` / `symexp(x)` used throughout for numerical stability with large-magnitude values.
- **AMP support**: Automatic mixed precision via `tools.Optimizer` with configurable precision (16/32).

### Paper (`paper_workspace/`)

- **`main.tex`** — Korean thesis using `kotex`. Builds with `latexmk -pdf`.
- Figures are `.png` files in the same directory, referenced by `main.tex`.

## Runtime Requirements

- CUDA GPU required (`device: 'cuda:0'`)
- MuJoCo with EGL rendering (`MUJOCO_GL=egl` set automatically for headless)
- PyTorch 2.4.1, gym 0.22.0
