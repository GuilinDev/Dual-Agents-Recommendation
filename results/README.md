# Results Directory

This directory stores experiment results.

## Output Files

After running experiments, you will find:
- `main_results.json` - Main comparison results
- `ablation_results.json` - Ablation study results
- `convergence_data.json` - Convergence curve data
- `pareto_fronts.json` - Pareto front coordinates

## Visualization

Results can be visualized using:
```bash
python src/run_experiments.py --visualize --results_dir results/
```
