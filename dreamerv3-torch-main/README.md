# LIVerse: Language & Image integrated uniVerse Model for Robot Foundation Models

A PyTorch implementation of **LIVerse**, a novel world foundation model (WFM) that unifies vision-language semantics and forward dynamics into a unified latent space to enhance the generality and scalability of model-based reinforcement learning. Unlike traditional imitation learning-based robot foundation models (RFMs), we introduce a PlaNet-style agent that leverages vision-language model (VLM) based reward in sampling-based model predictive control (MPC) manner.

## Overview

**LIVerse** addresses key limitations in current robot foundation models by:

- **Unified Latent Space**: Integrating vision-language semantics with forward dynamics in a single representation
- **Zero-Shot Policy Execution**: Extending zero-shot reward models to zero-shot policy models through MPC
- **Delta-Score Reward Function**: Measuring incremental similarity improvements to overcome cosine similarity limitations
- **Step-wise Planning**: Effective sub-task decomposition with adaptive goal switching
- **Reduced Hand-crafted Dependencies**: Minimizing reliance on manually designed reward functions

## Research Poster

![LIVerse Research Poster](./imgs/LIVerse_MBRL_Poster.png)

## Architecture

### 1. LIVerse World Foundation Model (`dreamer.py`)
- Unifies vision-language semantics and forward dynamics in a unified latent space
- Trains world model with VLM-based rewards for enhanced generality and scalability
- Generates imaginary trajectories that incorporate both visual and linguistic understanding

### 2. PlaNet-style MPC Agent
- **Basic MPC** (`planet_basic.py`): Single-goal planning with cosine similarity rewards
- **Step-wise MPC** (`planet_stepwise.py`): Multi-subtask planning with delta-score reward function

### 3. Delta-Score Reward Function
- Measures incremental similarity improvements rather than absolute similarity
- Addresses reward scale mismatch problem across sub-tasks
- Overcomes limitations of traditional cosine similarity rewards in complex tasks

### 4. VLM Integration (LIV)
- Language-Image Vision model for reward computation
- Supports both text and image goal specifications  
- Enables zero-shot transfer from reward models to policy models

## Installation

### Dependencies
```bash
pip install -r requirements.txt
```

### Additional Setup
The project requires:
- MetaWorld for robotic manipulation tasks
- LIV model for vision-language understanding
- MuJoCo for physics simulation

## Usage

### 1. World Model Training
Train the world model with VLM-based rewards:
```bash
# MetaWorld button press task
python dreamer.py --configs metaworld --task ML1_button-press-v2 --logdir ./logdir/button_press

# DMC Vision tasks
python dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/walker_walk
```

### 2. Zero-Shot MPC Planning

#### Basic Single-Goal Planning
```bash
python planet_basic.py --configs metaworld --task ML1_button-press-v2 \
    --logdir ./logdir/button_press \
    --planning_horizon 12 \
    --candidates 1000 \
    --eval_episodes 5 \
    --goal_image "success_frame_button" \
    --reward_form "similarity"
```

#### Multi-Subtask Planning
```bash
python planet_stepwise.py --configs metaworld --task ML1_button-press-v2 \
    --logdir ./logdir/button_press \
    --step_wise True \
    --subtask_dir "./subtasks" \
    --similarity_threshold 0.0012 \
    --plateau_patience 5 \
    --reward_form "difference"
```

### 3. Configuration Options

#### Planning Parameters
- `--planning_horizon`: MPC planning horizon (default: 12)
- `--candidates`: Number of action sequence candidates (default: 1000)
- `--top_candidates`: Top candidates for CEM update (default: 100)
- `--optimization_iters`: CEM optimization iterations (default: 10)

#### Reward Modes
- `--reward_form similarity`: Direct cosine similarity with target
- `--reward_form difference`: Delta-score reward based on incremental similarity improvements

#### Goal Specification
- `--goal_image`: Image file name for visual goals
- `--text_prompt`: Text description for language goals
- `--subtask_dir`: Directory containing sequential subtask images

## Key Features

### 1. World Foundation Model (WFM) Innovation
- **Unified Latent Space**: Integrates vision-language semantics with forward dynamics
- **Enhanced Generality**: Improved scalability compared to traditional model-based RL
- **Zero-Shot Transfer**: Extends zero-shot reward models to zero-shot policy models

### 2. Delta-Score Reward Function
- **Incremental Improvement Measurement**: Overcomes cosine similarity limitations
- **Reward Scale Matching**: Addresses mismatch problems across sub-tasks
- **Complex Task Handling**: Enables effective planning in multi-stage scenarios

### 3. Step-wise Planning & Sub-task Decomposition
- **Adaptive Goal Switching**: Automatic transition between sub-tasks
- **Plateau Detection**: Intelligent progress monitoring
- **Effective Decomposition**: Breaks complex tasks into manageable components

### 4. Reduced Hand-crafted Dependencies
- **Minimal Manual Design**: Significantly reduces reliance on hand-crafted rewards
- **Flexible Goal Specification**: Supports both image and text-based goals
- **Zero-Shot Generalization**: No additional training required for new tasks

## Experimental Results

### Benchmark Environment
| Environment | Task Type | Evaluation Focus |
|-------------|-----------|------------------|
| Meta-World | Robotic Manipulation | Zero-shot policy execution, Transfer learning, Sub-task decomposition |

### Key Findings
- **Strong Zero-Shot Policy Execution**: LIVerse-based MPC agent demonstrates effective zero-shot performance
- **Transfer Learning Across Diverse Tasks**: Successful generalization across various manipulation tasks and environments  
- **Effective Sub-task Decomposition**: Step-wise planning with delta-score rewards handles complex multi-stage tasks


## File Structure
```
├── dreamer.py              # LIVerse world foundation model training
├── planet_basic.py         # Single-goal PlaNet-style MPC planning
├── planet_stepwise.py      # Step-wise MPC with delta-score rewards
├── models.py              # World model architectures
├── tools.py               # Utility functions
├── metaworld_wrapper.py   # Meta-World environment wrapper
├── liv/                   # LIV vision-language model integration
└── configs.yaml           # Configuration parameters
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{liverse_rl,
  title={LIVerse: Language \& Image integrated uniVerse Model for Robot Foundation Models},
  author={Kyungseo Park},
  year={2025}
}
```

## Acknowledgments

This work builds upon:
- [DreamerV3](https://github.com/danijar/dreamerv3) - World model architecture
- [PlaNet](https://github.com/Kaixhin/PlaNet.git) - Model predictive control architecture
- [Meta-World](https://github.com/rlworkgroup/metaworld) - Robotic manipulation benchmark environment
- [LIV](https://github.com/penn-pal-lab/LIV.git) - Vision-Language Model for reward computation

## License

This project is licensed under the MIT License.
