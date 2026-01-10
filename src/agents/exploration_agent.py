"""
Exploration Agent for DualAgent-Rec.
Focuses on maximizing diversity and discovering novel items.
"""

import numpy as np
import random
from typing import List, Dict, Any, Optional
from .base_agent import BaseAgent, Individual

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class ExplorationAgent(BaseAgent):
    """
    Exploration Agent: Focuses on diversity and novelty.

    Uses unconstrained optimization with diversity-based selection.
    Primary objective: Maximize intra-list diversity and coverage.
    """

    def __init__(
        self,
        population_size: int = 100,
        num_objectives: int = 3,
        num_constraints: int = 3,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.2,  # Higher mutation for exploration
        diversity_weight: float = 0.6  # Weight for diversity in fitness
    ):
        super().__init__(
            population_size=population_size,
            num_objectives=num_objectives,
            num_constraints=num_constraints,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate
        )
        self.diversity_weight = diversity_weight
        self.candidate_items: List[str] = []
        self.k = 10

    def initialize_population(self, candidate_items: List[str], k: int = 10) -> None:
        """
        Initialize population with diverse solutions.
        Explicitly sample from different categories/clusters.
        """
        self.candidate_items = candidate_items
        self.k = k
        self.population = []

        for _ in range(self.population_size):
            # Random diverse sampling
            items = random.sample(candidate_items, min(k, len(candidate_items)))
            self.population.append(Individual(item_ids=items))

        self.generation = 0

    def evaluate_population(
        self,
        user_history: List[Dict],
        item_features: Dict[str, Dict],
        objectives_calculator: Any,
        constraints_handler: Any
    ) -> None:
        """
        Evaluate focusing on diversity objectives.
        Uses relaxed constraint handling to encourage exploration.
        """
        for individual in self.population:
            # Calculate objectives
            scores = objectives_calculator.calculate(
                recommended_items=individual.item_ids,
                user_history=user_history,
                item_features=item_features
            )
            individual.scores = np.array(scores)

            # Calculate constraint violations (but with relaxed handling)
            violations = constraints_handler.calculate_violations(
                recommended_items=individual.item_ids,
                item_features=item_features
            )
            individual.constraint_violations = np.array(violations)

            # Fitness: weighted combination favoring diversity
            # Score[1] = diversity, Score[2] = novelty
            diversity_score = individual.scores[1] * self.diversity_weight + \
                             individual.scores[2] * (1 - self.diversity_weight)

            # Soft constraint penalty (less strict than exploitation agent)
            penalty = 0.1 * individual.total_violation
            individual.fitness = diversity_score - penalty

        # Update archive (keep diverse solutions)
        self._update_diversity_archive()

    def _update_diversity_archive(self) -> None:
        """Update archive keeping diverse solutions."""
        # Sort by diversity fitness
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)

        # Keep top solutions ensuring diversity
        self.archive = []
        for ind in sorted_pop:
            if len(self.archive) >= self.population_size // 2:
                break
            # Check if significantly different from existing archive members
            is_diverse = True
            for arch_ind in self.archive:
                overlap = len(set(ind.item_ids) & set(arch_ind.item_ids)) / self.k
                if overlap > 0.7:  # Too similar
                    is_diverse = False
                    break
            if is_diverse:
                self.archive.append(ind)

    def evolve(self, num_offspring: int) -> List[Individual]:
        """
        Generate offspring using DE/rand/1 inspired strategy.
        Emphasizes exploration and diversity.
        """
        offspring = []

        for _ in range(num_offspring):
            # Select three random distinct individuals
            r1, r2, r3 = random.sample(self.population, 3)

            # DE/rand/1 style mutation
            child_items = []
            for i in range(self.k):
                rand_val = random.random()

                if rand_val < 0.33 and i < len(r1.item_ids):
                    item = r1.item_ids[i]
                elif rand_val < 0.66 and i < len(r2.item_ids):
                    item = r2.item_ids[i]
                elif i < len(r3.item_ids):
                    item = r3.item_ids[i]
                else:
                    item = random.choice(self.candidate_items)

                child_items.append(item)

            # Remove duplicates and ensure diversity
            child_items = list(dict.fromkeys(child_items))

            # Fill with random items (for diversity)
            while len(child_items) < self.k:
                available = [it for it in self.candidate_items if it not in child_items]
                if available:
                    child_items.append(random.choice(available))
                else:
                    break

            child = Individual(item_ids=child_items[:self.k])

            # Higher mutation rate for exploration
            child = self.mutate(child, self.candidate_items)

            offspring.append(child)

        return offspring

    def calculate_decision_space_diversity(self) -> float:
        """
        Calculate diversity in decision space (item combinations).
        Used for adaptive resource allocation.
        """
        if len(self.population) < 2:
            return 0.0

        total_distance = 0.0
        count = 0

        for i, ind1 in enumerate(self.population):
            for ind2 in self.population[i + 1:]:
                # Jaccard distance
                set1 = set(ind1.item_ids)
                set2 = set(ind2.item_ids)
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                distance = 1 - (intersection / union if union > 0 else 0)
                total_distance += distance
                count += 1

        return total_distance / count if count > 0 else 0.0

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get exploration-specific metrics."""
        base_metrics = super().get_performance_metrics()

        if self.population:
            # Add exploration-specific metrics
            base_metrics['diversity'] = np.mean([ind.scores[1] for ind in self.population])
            base_metrics['coverage'] = np.mean([ind.scores[2] for ind in self.population])
            base_metrics['decision_space_diversity'] = self.calculate_decision_space_diversity()

        return base_metrics
