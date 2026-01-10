"""
Experiment Runner for DualAgent-Rec.
Runs experiments with different configurations and baselines.

Key experiments:
1. Main comparison: DualAgent-Rec vs ablation variants
2. Ablation study: Effect of each component
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
import logging

# Add paths
sys.path.insert(0, os.path.dirname(__file__))

from dualagent_rec import DualAgentRec, DualAgentConfig
from data_utils import AmazonDataLoader, UserBehaviorProcessor
from evaluation import RecommendationMetrics, MultiObjectiveMetrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_amazon_data(data_dir: str, category: str = 'All_Beauty', max_reviews: int = 10000, n_users: int = 20) -> Tuple:
    """
    Load real Amazon data.

    Returns:
        Tuple of (candidate_items, user_histories, item_features, user_profiles)
    """
    logger.info(f"Loading Amazon {category} dataset...")

    loader = AmazonDataLoader(data_dir, category=category)

    # Download if needed
    loader.download_dataset()

    # Load and preprocess - scale max_reviews with n_users
    adjusted_max_reviews = max(max_reviews, n_users * 500)  # Ensure enough data
    df, metadata = loader.preprocess(
        min_user_interactions=5,
        min_item_interactions=3,
        max_reviews=adjusted_max_reviews
    )

    # Add compatibility fields to metadata
    for item_id, item_info in metadata.items():
        # Ensure both 'category' and 'main_category' exist
        if 'main_category' in item_info and 'category' not in item_info:
            item_info['category'] = item_info['main_category']
        elif 'category' in item_info and 'main_category' not in item_info:
            item_info['main_category'] = item_info['category']

        # Add seller_id if not present (use asin prefix as proxy)
        if 'seller_id' not in item_info:
            item_info['seller_id'] = f"seller_{hash(item_id) % 100}"

        # Add is_new flag (randomly for demo)
        if 'is_new' not in item_info:
            item_info['is_new'] = np.random.random() < 0.15

        # Add popularity based on interaction count
        if 'popularity' not in item_info:
            item_info['popularity'] = 0.5

    # Build user processor
    processor = UserBehaviorProcessor(df, metadata)

    # Get candidate items
    candidate_items = list(metadata.keys())

    # Calculate item popularity from interactions
    item_counts = df['item_id'].value_counts()
    max_count = item_counts.max() if len(item_counts) > 0 else 1
    for item_id in metadata:
        count = item_counts.get(item_id, 0)
        metadata[item_id]['popularity'] = count / max_count

    # Sample users for experiments
    user_ids = processor.sample_users(n_users=n_users, min_interactions=10)

    user_histories = {}
    user_profiles = {}
    for user_id in user_ids:
        user_histories[user_id] = processor.get_user_history(user_id, max_items=50)
        user_profiles[user_id] = processor.get_user_profile(user_id)

    logger.info(f"Loaded {len(candidate_items)} items, {len(user_ids)} users")

    return candidate_items, user_histories, metadata, user_profiles


def run_single_experiment(
    config: DualAgentConfig,
    candidate_items: List[str],
    user_history: List[Dict],
    item_features: Dict[str, Dict],
    user_profile: Dict[str, Any],
    experiment_name: str
) -> Dict[str, Any]:
    """Run a single experiment with given configuration."""
    logger.info(f"Running experiment: {experiment_name}")

    framework = DualAgentRec(config)

    start_time = datetime.now()
    best_solutions, metrics = framework.optimize(
        candidate_items=candidate_items,
        user_history=user_history,
        item_features=item_features,
        user_profile=user_profile
    )
    end_time = datetime.now()

    metrics['experiment_name'] = experiment_name
    metrics['runtime_seconds'] = (end_time - start_time).total_seconds()

    return metrics


def run_main_comparison(
    candidate_items: List[str],
    user_histories: Dict[str, List[Dict]],
    item_features: Dict[str, Dict],
    user_profiles: Dict[str, Dict],
    output_dir: str,
    use_llm: bool = False,
    num_runs: int = 3
) -> Dict[str, List[Dict]]:
    """
    Run main comparison experiments.

    Compares:
    1. DualAgent-Rec (full model)
    2. w/o LLM Coordinator (rule-based allocation)
    3. w/o Dual-Agent (single population)
    4. w/o Hard Constraints (soft penalty only)
    5. Random baseline
    """
    results = {}

    configs = {
        'DualAgent-Rec': DualAgentConfig(
            population_size=100,
            max_generations=50,
            recommendation_size=10,
            use_llm=use_llm,
            llm_model='qwen2.5:14b',
            fairness_threshold=0.6,
            seller_coverage_threshold=0.2,
            new_item_threshold=0.1,
        ),
        'w/o_LLM': DualAgentConfig(
            population_size=100,
            max_generations=50,
            recommendation_size=10,
            use_llm=False,  # Disable LLM
            fairness_threshold=0.6,
            seller_coverage_threshold=0.2,
            new_item_threshold=0.1,
        ),
        'w/o_Constraints': DualAgentConfig(
            population_size=100,
            max_generations=50,
            recommendation_size=10,
            use_llm=False,
            fairness_threshold=0.0,  # Disable constraints
            seller_coverage_threshold=0.0,
            new_item_threshold=0.0,
        ),
        'Single_Population': DualAgentConfig(
            population_size=200,  # Combined population
            max_generations=50,
            recommendation_size=10,
            use_llm=False,
            fairness_threshold=0.6,
            seller_coverage_threshold=0.2,
            new_item_threshold=0.1,
        ),
    }

    # Use all users for main comparison
    user_ids = list(user_histories.keys())
    logger.info(f"Running experiments on {len(user_ids)} users")

    for method_name, config in configs.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {method_name}")
        logger.info(f"{'='*50}")

        method_results = []
        for run in range(num_runs):
            logger.info(f"  Run {run+1}/{num_runs}")
            np.random.seed(RANDOM_SEED + run)

            # Run on each user and aggregate
            user_results = []
            for user_id in user_ids:
                try:
                    metrics = run_single_experiment(
                        config=config,
                        candidate_items=candidate_items,
                        user_history=user_histories[user_id],
                        item_features=item_features,
                        user_profile=user_profiles[user_id],
                        experiment_name=f"{method_name}_run{run+1}_{user_id}"
                    )
                    user_results.append(metrics)
                except Exception as e:
                    logger.error(f"  User {user_id} failed: {e}")
                    continue

            if user_results:
                # Aggregate across users
                aggregated = {
                    'hypervolume': np.mean([r.get('hypervolume', 0) for r in user_results]),
                    'spacing': np.mean([r.get('spacing', 0) for r in user_results]),
                    'pareto_size': int(np.mean([r.get('pareto_size', 0) for r in user_results])),
                    'avg_accuracy': np.mean([r.get('avg_accuracy', 0) for r in user_results]),
                    'avg_diversity': np.mean([r.get('avg_diversity', 0) for r in user_results]),
                    'avg_novelty': np.mean([r.get('avg_novelty', 0) for r in user_results]),
                    'feasibility_rate': np.mean([r.get('feasibility_rate', 0) for r in user_results]),
                    'runtime_seconds': np.sum([r.get('runtime_seconds', 0) for r in user_results]),
                    'experiment_name': f"{method_name}_run{run+1}",
                    'num_users': len(user_results),
                }
                method_results.append(aggregated)

        results[method_name] = method_results

        # Log summary
        if method_results:
            avg_hv = np.mean([r.get('hypervolume', 0) for r in method_results])
            avg_acc = np.mean([r.get('avg_accuracy', 0) for r in method_results])
            avg_div = np.mean([r.get('avg_diversity', 0) for r in method_results])
            avg_feasibility = np.mean([r.get('feasibility_rate', 0) for r in method_results])
            logger.info(f"  Avg HV: {avg_hv:.4f}, Acc: {avg_acc:.4f}, Div: {avg_div:.4f}, Feas: {avg_feasibility:.2%}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'main_comparison.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    return results


def run_ablation_study(
    candidate_items: List[str],
    user_histories: Dict[str, List[Dict]],
    item_features: Dict[str, Dict],
    user_profiles: Dict[str, Dict],
    output_dir: str
) -> Dict[str, Any]:
    """
    Run comprehensive ablation study.

    Studies:
    1. Population size effect
    2. Constraint threshold effect
    3. Mutation rate effect
    4. Generation count effect
    """
    logger.info("\n" + "="*50)
    logger.info("Running Ablation Study")
    logger.info("="*50)

    # Select one user for ablation study (faster)
    user_id = list(user_histories.keys())[0]
    user_history = user_histories[user_id]
    user_profile = user_profiles[user_id]

    ablation_configs = {
        # Population size study
        'Pop_50': DualAgentConfig(population_size=50, max_generations=30, use_llm=False),
        'Pop_100': DualAgentConfig(population_size=100, max_generations=30, use_llm=False),
        'Pop_200': DualAgentConfig(population_size=200, max_generations=30, use_llm=False),

        # Mutation rate study
        'Mutation_0.05': DualAgentConfig(population_size=100, max_generations=30, mutation_rate=0.05, use_llm=False),
        'Mutation_0.1': DualAgentConfig(population_size=100, max_generations=30, mutation_rate=0.1, use_llm=False),
        'Mutation_0.2': DualAgentConfig(population_size=100, max_generations=30, mutation_rate=0.2, use_llm=False),

        # Constraint threshold study
        'Strict_Constraints': DualAgentConfig(
            population_size=100, max_generations=30, use_llm=False,
            fairness_threshold=0.8, seller_coverage_threshold=0.4, new_item_threshold=0.2
        ),
        'Normal_Constraints': DualAgentConfig(
            population_size=100, max_generations=30, use_llm=False,
            fairness_threshold=0.6, seller_coverage_threshold=0.2, new_item_threshold=0.1
        ),
        'Relaxed_Constraints': DualAgentConfig(
            population_size=100, max_generations=30, use_llm=False,
            fairness_threshold=0.4, seller_coverage_threshold=0.1, new_item_threshold=0.05
        ),

        # Generation count study
        'Gen_20': DualAgentConfig(population_size=100, max_generations=20, use_llm=False),
        'Gen_50': DualAgentConfig(population_size=100, max_generations=50, use_llm=False),
        'Gen_100': DualAgentConfig(population_size=100, max_generations=100, use_llm=False),
    }

    results = {}
    for name, config in ablation_configs.items():
        logger.info(f"Running ablation: {name}")
        try:
            metrics = run_single_experiment(
                config=config,
                candidate_items=candidate_items,
                user_history=user_history,
                item_features=item_features,
                user_profile=user_profile,
                experiment_name=name
            )
            results[name] = metrics
            logger.info(f"  HV: {metrics.get('hypervolume', 0):.4f}, Acc: {metrics.get('avg_accuracy', 0):.4f}")
        except Exception as e:
            logger.error(f"Ablation {name} failed: {e}")

    # Save results
    output_path = os.path.join(output_dir, 'ablation_study.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Ablation results saved to {output_path}")

    return results


def generate_results_table(results: Dict[str, List[Dict]]) -> str:
    """Generate LaTeX-ready results table."""
    table = "\\begin{table}[h]\n\\centering\n"
    table += "\\caption{Main comparison results on Amazon dataset}\n"
    table += "\\label{tab:main_results}\n"
    table += "\\begin{tabular}{lcccc}\n\\toprule\n"
    table += "Method & HV $\\uparrow$ & Accuracy $\\uparrow$ & Diversity $\\uparrow$ & Feasibility $\\uparrow$ \\\\ \\midrule\n"

    for method, runs in results.items():
        if not runs:
            continue
        avg_hv = np.mean([r.get('hypervolume', 0) for r in runs])
        std_hv = np.std([r.get('hypervolume', 0) for r in runs])
        avg_acc = np.mean([r.get('avg_accuracy', 0) for r in runs])
        avg_div = np.mean([r.get('avg_diversity', 0) for r in runs])
        avg_feas = np.mean([r.get('feasibility_rate', 0) for r in runs])

        method_display = method.replace('_', ' ')
        if method == 'DualAgent-Rec':
            method_display = "\\textbf{DualAgent-Rec (Ours)}"

        table += f"{method_display} & "
        table += f"{avg_hv:.4f}$\\pm${std_hv:.4f} & "
        table += f"{avg_acc:.4f} & "
        table += f"{avg_div:.4f} & "
        table += f"{avg_feas:.2%} \\\\\n"

    table += "\\bottomrule\n\\end{tabular}\n"
    table += "\\end{table}"

    return table


def generate_ablation_table(results: Dict[str, Dict]) -> str:
    """Generate LaTeX table for ablation study."""
    table = "\\begin{table}[h]\n\\centering\n"
    table += "\\caption{Ablation study results}\n"
    table += "\\label{tab:ablation}\n"
    table += "\\begin{tabular}{lccccc}\n\\toprule\n"
    table += "Setting & HV & Accuracy & Diversity & Novelty & Runtime(s) \\\\ \\midrule\n"

    for name, metrics in results.items():
        if not metrics:
            continue
        table += f"{name.replace('_', ' ')} & "
        table += f"{metrics.get('hypervolume', 0):.4f} & "
        table += f"{metrics.get('avg_accuracy', 0):.4f} & "
        table += f"{metrics.get('avg_diversity', 0):.4f} & "
        table += f"{metrics.get('avg_novelty', 0):.4f} & "
        table += f"{metrics.get('runtime_seconds', 0):.1f} \\\\\n"

    table += "\\bottomrule\n\\end{tabular}\n"
    table += "\\end{table}"

    return table


def main():
    parser = argparse.ArgumentParser(description='Run DualAgent-Rec experiments')
    parser.add_argument('--data_dir', type=str, default='../data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='../experiments/results', help='Output directory')
    parser.add_argument('--categories', type=str, nargs='+',
                       default=['All_Beauty'],
                       help='Amazon dataset categories (can specify multiple)')
    parser.add_argument('--n_users', type=int, default=20, help='Number of users per category')
    parser.add_argument('--use_llm', action='store_true', help='Use LLM coordinator')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of runs per method')
    parser.add_argument('--max_reviews', type=int, default=20000, help='Max reviews to load')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'main', 'ablation', 'quick'],
                       help='Which experiments to run')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load real Amazon data from multiple categories
    all_candidate_items = []
    all_user_histories = {}
    all_item_features = {}
    all_user_profiles = {}

    for category in args.categories:
        logger.info(f"\n{'='*50}")
        logger.info(f"Loading category: {category}")
        logger.info(f"{'='*50}")
        try:
            candidate_items, user_histories, item_features, user_profiles = load_amazon_data(
                args.data_dir,
                category=category,
                max_reviews=args.max_reviews,
                n_users=args.n_users
            )
            # Merge data from this category (with consistent prefixing)
            all_candidate_items.extend([f"{category}_{item_id}" for item_id in candidate_items])
            for user_id, history in user_histories.items():
                # Update item_ids in history to have prefix
                prefixed_history = []
                for h in history:
                    h_copy = h.copy()
                    if 'item_id' in h_copy:
                        h_copy['item_id'] = f"{category}_{h_copy['item_id']}"
                    prefixed_history.append(h_copy)
                all_user_histories[f"{category}_{user_id}"] = prefixed_history
            for item_id, features in item_features.items():
                features['source_category'] = category
                all_item_features[f"{category}_{item_id}"] = features
            for user_id, profile in user_profiles.items():
                all_user_profiles[f"{category}_{user_id}"] = profile
            logger.info(f"Loaded {len(candidate_items)} items, {len(user_histories)} users from {category}")
        except Exception as e:
            logger.error(f"Failed to load {category}: {e}")
            continue

    # Use merged data
    candidate_items = list(set(all_candidate_items))
    user_histories = all_user_histories
    item_features = all_item_features
    user_profiles = all_user_profiles

    logger.info(f"\nTotal data loaded: {len(candidate_items)} items, {len(user_histories)} users across {len(args.categories)} categories")

    # Run experiments
    if args.experiment in ['all', 'main']:
        results = run_main_comparison(
            candidate_items=candidate_items,
            user_histories=user_histories,
            item_features=item_features,
            user_profiles=user_profiles,
            output_dir=args.output_dir,
            use_llm=args.use_llm,
            num_runs=args.num_runs
        )

        # Generate table
        table = generate_results_table(results)
        table_path = os.path.join(args.output_dir, 'results_table.tex')
        with open(table_path, 'w') as f:
            f.write(table)
        logger.info(f"LaTeX table saved to {table_path}")

    if args.experiment in ['all', 'ablation']:
        ablation_results = run_ablation_study(
            candidate_items=candidate_items,
            user_histories=user_histories,
            item_features=item_features,
            user_profiles=user_profiles,
            output_dir=args.output_dir
        )

        # Generate ablation table
        ablation_table = generate_ablation_table(ablation_results)
        ablation_table_path = os.path.join(args.output_dir, 'ablation_table.tex')
        with open(ablation_table_path, 'w') as f:
            f.write(ablation_table)
        logger.info(f"Ablation table saved to {ablation_table_path}")

    if args.experiment == 'quick':
        # Quick test run with one user
        logger.info("Running quick test...")
        user_id = list(user_histories.keys())[0]

        config = DualAgentConfig(
            population_size=30,
            max_generations=10,
            use_llm=False,
        )
        metrics = run_single_experiment(
            config=config,
            candidate_items=candidate_items,
            user_history=user_histories[user_id],
            item_features=item_features,
            user_profile=user_profiles[user_id],
            experiment_name='quick_test'
        )
        print("\n=== Quick Test Results ===")
        print(f"Hypervolume: {metrics.get('hypervolume', 0):.4f}")
        print(f"Accuracy: {metrics.get('avg_accuracy', 0):.4f}")
        print(f"Diversity: {metrics.get('avg_diversity', 0):.4f}")
        print(f"Novelty: {metrics.get('avg_novelty', 0):.4f}")
        print(f"Pareto size: {metrics.get('pareto_size', 0)}")
        print(f"Feasibility: {metrics.get('feasibility_rate', 0):.2%}")
        print(f"Runtime: {metrics.get('runtime_seconds', 0):.2f}s")

    logger.info("\nAll experiments completed!")


if __name__ == "__main__":
    main()
