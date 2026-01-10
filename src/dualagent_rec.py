"""
DualAgent-Rec: LLM-Coordinated Dual-Agent Framework for
Constrained Multi-Objective E-commerce Recommendation

Main framework integrating all components.
"""

import sys
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
import json
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

from agents import ExploitationAgent, ExplorationAgent, Individual
from constraints import ConstraintHandler, ConstraintConfig, AdaptiveConstraintHandler
from evaluation import ObjectivesCalculator, RecommendationMetrics, MultiObjectiveMetrics
from llm_coordinator import LLMCoordinator, CoordinatorConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


@dataclass
class DualAgentConfig:
    """Configuration for DualAgent-Rec framework."""
    # Population settings
    population_size: int = 100
    max_generations: int = 200
    recommendation_size: int = 10

    # Agent settings
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1

    # Constraint settings
    fairness_threshold: float = 0.7
    seller_coverage_threshold: float = 0.3
    new_item_threshold: float = 0.1

    # LLM settings
    use_llm: bool = True
    llm_model: str = 'qwen2.5:72b'
    llm_update_frequency: int = 10

    # Output settings
    output_dir: str = './results'
    save_history: bool = True


class DualAgentRec:
    """
    DualAgent-Rec Framework.

    Combines:
    1. Exploitation Agent (accuracy-focused)
    2. Exploration Agent (diversity-focused)
    3. LLM Coordinator (resource allocation)
    4. Adaptive Constraint Handler
    """

    def __init__(self, config: Optional[DualAgentConfig] = None):
        """
        Initialize DualAgent-Rec.

        Args:
            config: Framework configuration
        """
        self.config = config or DualAgentConfig()

        # Initialize components
        self._init_agents()
        self._init_coordinator()
        self._init_constraint_handler()
        self._init_objectives_calculator()

        # State
        self.generation = 0
        self.history: List[Dict[str, Any]] = []
        self.best_solutions: List[Individual] = []

    def _init_agents(self):
        """Initialize dual agents."""
        self.exploitation_agent = ExploitationAgent(
            population_size=self.config.population_size,
            crossover_rate=self.config.crossover_rate,
            mutation_rate=self.config.mutation_rate
        )

        self.exploration_agent = ExplorationAgent(
            population_size=self.config.population_size,
            crossover_rate=self.config.crossover_rate,
            mutation_rate=self.config.mutation_rate * 2  # Higher mutation for exploration
        )

    def _init_coordinator(self):
        """Initialize LLM coordinator."""
        coord_config = CoordinatorConfig(
            model_name=self.config.llm_model,
            update_frequency=self.config.llm_update_frequency,
            use_llm=self.config.use_llm
        )
        self.coordinator = LLMCoordinator(coord_config)

    def _init_constraint_handler(self):
        """Initialize constraint handler."""
        constraint_config = ConstraintConfig(
            fairness_threshold=self.config.fairness_threshold,
            seller_coverage_threshold=self.config.seller_coverage_threshold,
            new_item_threshold=self.config.new_item_threshold
        )
        self.constraint_handler = AdaptiveConstraintHandler(constraint_config)

    def _init_objectives_calculator(self):
        """Initialize objectives calculator."""
        self.objectives_calculator = ObjectivesCalculator()

    def optimize(
        self,
        candidate_items: List[str],
        user_history: List[Dict],
        item_features: Dict[str, Dict],
        item_embeddings: Optional[Dict[str, np.ndarray]] = None,
        item_popularity: Optional[Dict[str, float]] = None,
        user_profile: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Individual], Dict[str, Any]]:
        """
        Run multi-objective optimization.

        Args:
            candidate_items: List of candidate item IDs
            user_history: User's interaction history
            item_features: Item ID -> features mapping
            item_embeddings: Optional item embeddings
            item_popularity: Optional item popularity scores
            user_profile: Optional user profile

        Returns:
            Tuple of (Pareto-optimal solutions, optimization metrics)
        """
        logger.info("Starting DualAgent-Rec optimization...")

        # Update objectives calculator
        if item_embeddings:
            self.objectives_calculator.item_embeddings = item_embeddings
        if item_popularity:
            self.objectives_calculator.item_popularity = item_popularity

        # Initialize populations
        self._initialize_populations(candidate_items)

        # Main optimization loop
        for gen in range(self.config.max_generations):
            self.generation = gen

            # Evaluate populations
            self._evaluate_populations(user_history, item_features)

            # Get agent metrics
            exploit_metrics = self.exploitation_agent.get_performance_metrics()
            explore_metrics = self.exploration_agent.get_performance_metrics()

            # Get constraint metrics
            constraint_metrics = self._get_constraint_metrics()

            # Get resource allocation from coordinator
            exploitation_ratio, reasoning = self.coordinator.get_resource_allocation(
                exploitation_metrics=exploit_metrics,
                exploration_metrics=explore_metrics,
                constraint_metrics=constraint_metrics,
                current_generation=gen,
                max_generations=self.config.max_generations,
                user_profile=user_profile
            )

            # Calculate offspring counts based on allocation
            total_offspring = self.config.population_size
            exploit_offspring = int(total_offspring * exploitation_ratio)
            explore_offspring = total_offspring - exploit_offspring

            # Generate offspring
            exploit_children = self.exploitation_agent.evolve(exploit_offspring)
            explore_children = self.exploration_agent.evolve(explore_offspring)

            # Cross-population breeding (knowledge transfer)
            transfer_children = self._cross_population_breeding(
                self.exploitation_agent.population,
                self.exploration_agent.population,
                num_children=min(10, total_offspring // 5)
            )

            # Combine and select
            combined_exploit = self.exploitation_agent.population + exploit_children + transfer_children
            combined_explore = self.exploration_agent.population + explore_children + transfer_children

            self.exploitation_agent.population = self.exploitation_agent.environmental_selection(combined_exploit)
            self.exploration_agent.population = self.exploration_agent.environmental_selection(combined_explore)

            # Update constraint epsilon
            self.constraint_handler.update_epsilon(constraint_metrics.get('overall_feasibility', 0.5))

            # Update best solutions
            self._update_best_solutions()

            # Log progress
            if gen % 10 == 0:
                self._log_progress(gen, exploit_metrics, explore_metrics, constraint_metrics, reasoning)

            # Save history
            if self.config.save_history:
                self._save_generation_history(gen, exploit_metrics, explore_metrics, constraint_metrics)

        # Final results
        final_metrics = self._compute_final_metrics()
        logger.info(f"Optimization complete. Found {len(self.best_solutions)} Pareto-optimal solutions.")

        return self.best_solutions, final_metrics

    def _initialize_populations(self, candidate_items: List[str]):
        """Initialize both agent populations."""
        self.exploitation_agent.initialize_population(
            candidate_items,
            k=self.config.recommendation_size
        )
        self.exploration_agent.initialize_population(
            candidate_items,
            k=self.config.recommendation_size
        )

    def _evaluate_populations(self, user_history: List[Dict], item_features: Dict[str, Dict]):
        """Evaluate both populations."""
        self.exploitation_agent.evaluate_population(
            user_history,
            item_features,
            self.objectives_calculator,
            self.constraint_handler
        )
        self.exploration_agent.evaluate_population(
            user_history,
            item_features,
            self.objectives_calculator,
            self.constraint_handler
        )

    def _get_constraint_metrics(self) -> Dict[str, float]:
        """Get constraint satisfaction metrics from both populations."""
        all_individuals = self.exploitation_agent.population + self.exploration_agent.population

        if not all_individuals:
            return {}

        feasible_count = sum(1 for ind in all_individuals if ind.is_feasible)
        total_count = len(all_individuals)

        avg_violations = np.mean([ind.constraint_violations for ind in all_individuals], axis=0)

        return {
            'overall_feasibility': feasible_count / total_count,
            'fairness_violation': avg_violations[0] if len(avg_violations) > 0 else 0,
            'seller_violation': avg_violations[1] if len(avg_violations) > 1 else 0,
            'new_item_violation': avg_violations[2] if len(avg_violations) > 2 else 0,
        }

    def _cross_population_breeding(
        self,
        pop1: List[Individual],
        pop2: List[Individual],
        num_children: int
    ) -> List[Individual]:
        """
        Cross-population breeding for knowledge transfer.

        Combines good solutions from both populations.
        """
        children = []

        for _ in range(num_children):
            # Select parents from different populations
            parent1 = self.exploitation_agent.tournament_selection(pop1)
            parent2 = self.exploration_agent.tournament_selection(pop2)

            # Crossover
            child1, child2 = self.exploitation_agent.crossover(parent1, parent2)
            children.extend([child1, child2])

        return children[:num_children]

    def _update_best_solutions(self):
        """Update Pareto-optimal solutions from both populations."""
        all_feasible = [
            ind for ind in
            self.exploitation_agent.population + self.exploration_agent.population
            if ind.is_feasible
        ]

        if not all_feasible:
            # Keep best infeasible if no feasible found
            all_individuals = self.exploitation_agent.population + self.exploration_agent.population
            all_feasible = sorted(all_individuals, key=lambda x: x.total_violation)[:10]

        # Non-dominated sorting to get Pareto front
        fronts = self.exploitation_agent.non_dominated_sort(all_feasible)
        if fronts:
            self.best_solutions = fronts[0][:self.config.population_size]

    def _log_progress(
        self,
        generation: int,
        exploit_metrics: Dict,
        explore_metrics: Dict,
        constraint_metrics: Dict,
        reasoning: str
    ):
        """Log optimization progress."""
        logger.info(
            f"Gen {generation}: "
            f"Feasibility={constraint_metrics.get('overall_feasibility', 0):.2%}, "
            f"NDCG={exploit_metrics.get('ndcg', 0):.4f}, "
            f"Diversity={explore_metrics.get('diversity', 0):.4f}, "
            f"Pareto={len(self.best_solutions)}"
        )

    def _save_generation_history(
        self,
        generation: int,
        exploit_metrics: Dict,
        explore_metrics: Dict,
        constraint_metrics: Dict
    ):
        """Save generation history for analysis."""
        self.history.append({
            'generation': generation,
            'exploitation_metrics': exploit_metrics,
            'exploration_metrics': explore_metrics,
            'constraint_metrics': constraint_metrics,
            'pareto_size': len(self.best_solutions),
            'coordinator_summary': self.coordinator.get_coordination_summary(),
        })

    def _compute_final_metrics(self) -> Dict[str, Any]:
        """Compute final optimization metrics."""
        if not self.best_solutions:
            return {}

        # Extract Pareto front scores
        pareto_scores = [ind.scores for ind in self.best_solutions]

        # Compute multi-objective metrics
        reference_point = np.array([1.0, 1.0, 1.0])  # Ideal point
        hv = MultiObjectiveMetrics.hypervolume(pareto_scores, reference_point)
        spacing = MultiObjectiveMetrics.spacing(pareto_scores)

        # Compute recommendation metrics
        avg_scores = np.mean(pareto_scores, axis=0)

        return {
            'hypervolume': hv,
            'spacing': spacing,
            'pareto_size': len(self.best_solutions),
            'avg_accuracy': float(avg_scores[0]),
            'avg_diversity': float(avg_scores[1]),
            'avg_novelty': float(avg_scores[2]),
            'feasibility_rate': sum(1 for ind in self.best_solutions if ind.is_feasible) / len(self.best_solutions),
            'coordinator_summary': self.coordinator.get_coordination_summary(),
            'total_generations': self.generation + 1,
        }

    def get_recommendation(self, strategy: str = 'balanced') -> Optional[Individual]:
        """
        Get final recommendation from Pareto front.

        Args:
            strategy: Selection strategy
                - 'balanced': Best average score
                - 'accuracy': Best accuracy
                - 'diversity': Best diversity
                - 'novelty': Best novelty

        Returns:
            Selected recommendation solution
        """
        if not self.best_solutions:
            return None

        if strategy == 'balanced':
            return max(self.best_solutions, key=lambda x: np.mean(x.scores))
        elif strategy == 'accuracy':
            return max(self.best_solutions, key=lambda x: x.scores[0])
        elif strategy == 'diversity':
            return max(self.best_solutions, key=lambda x: x.scores[1])
        elif strategy == 'novelty':
            return max(self.best_solutions, key=lambda x: x.scores[2])
        else:
            return self.best_solutions[0]

    def save_results(self, output_path: str):
        """Save optimization results to file."""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        results = {
            'config': {
                'population_size': self.config.population_size,
                'max_generations': self.config.max_generations,
                'recommendation_size': self.config.recommendation_size,
                'use_llm': self.config.use_llm,
                'llm_model': self.config.llm_model,
            },
            'final_metrics': self._compute_final_metrics(),
            'history': self.history,
            'best_solutions': [
                {
                    'items': ind.item_ids,
                    'scores': ind.scores.tolist(),
                    'violations': ind.constraint_violations.tolist(),
                    'is_feasible': ind.is_feasible,
                }
                for ind in self.best_solutions
            ],
            'timestamp': datetime.now().isoformat(),
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")


def run_demo():
    """Run a simple demo of DualAgent-Rec."""
    import random

    # Create synthetic data
    num_items = 500
    num_history = 50

    candidate_items = [f"item_{i}" for i in range(num_items)]
    categories = ['Electronics', 'Computers', 'Phone', 'Camera', 'Audio', 'Gaming']
    sellers = [f"seller_{i}" for i in range(50)]

    item_features = {
        item_id: {
            'category': random.choice(categories),
            'seller_id': random.choice(sellers),
            'is_new': random.random() < 0.2,
            'popularity': random.random(),
        }
        for item_id in candidate_items
    }

    user_history = [
        {
            'item_id': random.choice(candidate_items),
            'category': random.choice(categories),
            'rating': random.randint(1, 5),
        }
        for _ in range(num_history)
    ]

    user_profile = {
        'interaction_count': num_history,
        'category_diversity': 0.7,
        'avg_rating': 4.2,
    }

    # Run optimization
    config = DualAgentConfig(
        population_size=50,
        max_generations=30,
        recommendation_size=10,
        use_llm=False,  # Disable LLM for quick demo
    )

    framework = DualAgentRec(config)

    best_solutions, metrics = framework.optimize(
        candidate_items=candidate_items,
        user_history=user_history,
        item_features=item_features,
        user_profile=user_profile
    )

    print("\n=== Demo Results ===")
    print(f"Pareto-optimal solutions: {len(best_solutions)}")
    print(f"Hypervolume: {metrics.get('hypervolume', 0):.4f}")
    print(f"Average accuracy: {metrics.get('avg_accuracy', 0):.4f}")
    print(f"Average diversity: {metrics.get('avg_diversity', 0):.4f}")
    print(f"Feasibility rate: {metrics.get('feasibility_rate', 0):.2%}")

    # Get recommendation
    rec = framework.get_recommendation('balanced')
    if rec:
        print(f"\nRecommended items: {rec.item_ids[:5]}...")
        print(f"Scores: {rec.scores}")


if __name__ == "__main__":
    run_demo()
