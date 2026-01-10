# DualAgent-Rec: LLM-Coordinated Dual-Agent Framework for Constrained Multi-Objective E-commerce Recommendation

This repository contains the implementation of **DualAgent-Rec**, an LLM-coordinated dual-agent framework for constrained multi-objective e-commerce recommendation.

## Overview

DualAgent-Rec provides:
- **Dual-Agent Architecture**: Two specialized evolutionary agents for exploration and exploitation
  - Exploitation Agent: CDP-based selection with DE/pbest/1 variation
  - Exploration Agent: Unconstrained Pareto dominance with doubled mutation rate
- **LLM-Based Coordination**: Intelligent resource allocation using Qwen2.5-14B
- **Adaptive Constraint Handling**: Self-calibrating ε-relaxation for hard constraints
  - Category Fairness (Gini coefficient)
  - Seller Coverage
  - New Item Exposure
- **Multi-Objective Optimization**: Balancing accuracy, diversity, and novelty

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/DualAgent-Rec.git
cd DualAgent-Rec

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama and pull models
# See https://ollama.ai for installation instructions
ollama pull qwen2.5:14b
```

## Project Structure

```
DualAgent-Rec/
├── src/
│   ├── agents/
│   │   ├── base_agent.py           # Base agent class
│   │   ├── exploitation_agent.py   # CDP-based exploitation agent
│   │   └── exploration_agent.py    # Pareto-based exploration agent
│   ├── llm_coordinator/
│   │   └── coordinator.py          # LLM-based resource coordinator
│   ├── constraints/
│   │   └── constraint_handler.py   # Constraint handling with ε-relaxation
│   ├── evaluation/
│   │   └── objectives.py           # Multi-objective evaluation metrics
│   ├── dualagent_rec.py            # Main framework implementation
│   └── run_experiments.py          # Experiment runner
├── data/                           # Dataset directory
├── results/                        # Experiment results
├── requirements.txt
└── README.md
```

## Usage

### Running Experiments

```bash
# Run main experiments with default settings
python src/run_experiments.py \
    --n_users 100 \
    --categories All_Beauty,Electronics,Clothing_Shoes_and_Jewelry \
    --output_dir results/

# Run with specific configuration
python src/run_experiments.py \
    --n_users 50 \
    --pop_size 100 \
    --max_gen 50 \
    --llm_model qwen2.5:14b
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--n_users` | Number of users per category | 100 |
| `--categories` | Amazon product categories | All_Beauty |
| `--pop_size` | Population size | 100 |
| `--max_gen` | Maximum generations | 50 |
| `--llm_model` | LLM model for coordinator | qwen2.5:14b |
| `--coord_interval` | Coordination interval | 10 |
| `--epsilon_decay` | ε decay rate | 0.8 |

## Framework Components

### 1. Exploitation Agent
Maintains a population optimized using the Constraint Domination Principle (CDP). Uses DE/pbest/1 variation to focus search around best performers.

### 2. Exploration Agent
Maintains a population with doubled mutation rate, using unconstrained Pareto dominance for selection. Discovers diverse solutions in unexplored regions.

### 3. LLM Coordinator
Analyzes optimization state and dynamically allocates resources (α ratio) between agents based on:
- Optimization progress (hypervolume improvement)
- Constraint satisfaction status
- Population diversity metrics

### 4. Adaptive ε-Constraint Handling
Gradually tightens constraints using: `ε_t = ε_0 × γ^(t/T_max)`

## Constraints

| Constraint | Description | Default Threshold |
|------------|-------------|-------------------|
| Category Fairness | Gini coefficient ≤ θ | θ_fair = 0.6 |
| Seller Coverage | Unique sellers ≥ θ × k | θ_seller = 0.2 |
| New Item Exposure | Items from last 30 days ≥ θ × k | θ_new = 0.1 |

## Metrics

- **Hypervolume (HV)**: Overall Pareto front quality
- **NDCG@10**: Ranking quality using held-out interactions
- **Diversity**: Average intra-list distance
- **Feasibility Rate**: Percentage of solutions satisfying all constraints

## Dataset

We use [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) dataset. The data loader automatically downloads and processes the specified categories.

Supported categories include:
- All_Beauty
- Electronics
- Clothing_Shoes_and_Jewelry
- And more...

## Results

Our experiments demonstrate:
- **100% constraint satisfaction** across all configurations
- **4.3% relative HV improvement** over single-population baselines
- Adaptive LLM coordination provides interpretable resource allocation decisions

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{dualagent2026,
  title={DualAgent-Rec: LLM-Coordinated Dual-Agent Framework for Constrained Multi-Objective E-commerce Recommendation},
  author={Anonymous},
  booktitle={Proceedings of the WWW 2026 Workshop on LLM \& Agents for Recommendation Systems},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon:
- NSGA-II for multi-objective optimization
- Adaptive ε-constraint methods for constraint handling
- Ollama for local LLM deployment
