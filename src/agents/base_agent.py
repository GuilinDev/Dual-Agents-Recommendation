"""
Base Agent class for DualAgent-Rec framework.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import random

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


@dataclass
class Individual:
    """Represents a recommendation solution (individual in population)."""
    item_ids: List[str]  # List of recommended item IDs
    scores: np.ndarray = field(default_factory=lambda: np.array([]))  # Objective scores [f1, f2, f3]
    constraint_violations: np.ndarray = field(default_factory=lambda: np.array([]))  # Constraint violations
    fitness: float = 0.0
    rank: int = 0
    crowding_distance: float = 0.0

    def __post_init__(self):
        if len(self.scores) == 0:
            self.scores = np.zeros(3)
        if len(self.constraint_violations) == 0:
            self.constraint_violations = np.zeros(3)

    @property
    def is_feasible(self) -> bool:
        """Check if solution satisfies all constraints."""
        return np.all(self.constraint_violations <= 0)

    @property
    def total_violation(self) -> float:
        """Total constraint violation."""
        return np.sum(np.maximum(0, self.constraint_violations))

    def dominates(self, other: 'Individual', use_constraints: bool = True) -> bool:
        """
        Check if this individual dominates another.

        Constraint-Domination Principle (CDP):
        1. Feasible dominates infeasible
        2. Between feasible: Pareto dominance
        3. Between infeasible: lower violation dominates
        """
        if use_constraints:
            # CDP comparison
            if self.is_feasible and not other.is_feasible:
                return True
            if not self.is_feasible and other.is_feasible:
                return False
            if not self.is_feasible and not other.is_feasible:
                return self.total_violation < other.total_violation

        # Pareto dominance (all objectives to maximize)
        better_or_equal = np.all(self.scores >= other.scores)
        strictly_better = np.any(self.scores > other.scores)
        return better_or_equal and strictly_better


class BaseAgent(ABC):
    """Abstract base class for recommendation agents."""

    def __init__(
        self,
        population_size: int = 100,
        num_objectives: int = 3,
        num_constraints: int = 3,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1
    ):
        """
        Initialize base agent.

        Args:
            population_size: Size of the population
            num_objectives: Number of objectives (default 3: CTR, diversity, novelty)
            num_constraints: Number of constraints (default 3: fairness, seller coverage, new items)
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
        """
        self.population_size = population_size
        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        self.population: List[Individual] = []
        self.archive: List[Individual] = []  # Pareto archive
        self.generation = 0

    @abstractmethod
    def initialize_population(self, candidate_items: List[str], k: int) -> None:
        """Initialize population with random solutions."""
        pass

    @abstractmethod
    def evaluate_population(
        self,
        user_history: List[Dict],
        item_features: Dict[str, Dict],
        objectives_calculator: 'ObjectivesCalculator',
        constraints_handler: 'ConstraintHandler'
    ) -> None:
        """Evaluate all individuals in population."""
        pass

    @abstractmethod
    def evolve(self, num_offspring: int) -> List[Individual]:
        """Generate offspring through genetic operators."""
        pass

    def non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """
        Fast non-dominated sorting.

        Returns:
            List of fronts (front 0 is best)
        """
        n = len(population)
        if n == 0:
            return []

        domination_count = [0] * n  # Number of solutions that dominate this
        dominated_solutions = [[] for _ in range(n)]  # Solutions dominated by this
        fronts = [[]]

        # Calculate domination relationships
        for i in range(n):
            for j in range(i + 1, n):
                if population[i].dominates(population[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif population[j].dominates(population[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        # Build index mapping
        ind_to_idx = {id(ind): i for i, ind in enumerate(population)}

        # Find first front
        first_front_indices = []
        for i in range(n):
            if domination_count[i] == 0:
                population[i].rank = 0
                fronts[0].append(population[i])
                first_front_indices.append(i)

        # Generate subsequent fronts
        current_front = 0
        current_indices = first_front_indices

        while current_indices:
            next_front = []
            next_indices = []
            for idx in current_indices:
                for j in dominated_solutions[idx]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        population[j].rank = current_front + 1
                        next_front.append(population[j])
                        next_indices.append(j)
            current_front += 1
            if next_front:
                fronts.append(next_front)
            current_indices = next_indices

        return fronts

    def calculate_crowding_distance(self, front: List[Individual]) -> None:
        """Calculate crowding distance for individuals in a front."""
        n = len(front)
        if n == 0:
            return

        for ind in front:
            ind.crowding_distance = 0.0

        for m in range(self.num_objectives):
            # Sort by objective m
            front.sort(key=lambda x: x.scores[m])

            # Boundary solutions get infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            # Calculate range
            obj_range = front[-1].scores[m] - front[0].scores[m]
            if obj_range == 0:
                continue

            # Calculate crowding distance
            for i in range(1, n - 1):
                front[i].crowding_distance += (
                    (front[i + 1].scores[m] - front[i - 1].scores[m]) / obj_range
                )

    def tournament_selection(self, population: List[Individual], k: int = 2) -> Individual:
        """
        Binary tournament selection using constraint-domination.
        """
        candidates = random.sample(population, min(k, len(population)))
        best = candidates[0]
        for candidate in candidates[1:]:
            # CDP comparison
            if candidate.dominates(best):
                best = candidate
            elif not best.dominates(candidate):
                # Neither dominates, use crowding distance
                if candidate.crowding_distance > best.crowding_distance:
                    best = candidate
        return best

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Uniform crossover for recommendation lists.
        """
        if random.random() > self.crossover_rate:
            return Individual(item_ids=parent1.item_ids.copy()), Individual(item_ids=parent2.item_ids.copy())

        # Use minimum length to avoid index errors
        k = min(len(parent1.item_ids), len(parent2.item_ids))
        child1_items = []
        child2_items = []

        for i in range(k):
            if random.random() < 0.5:
                child1_items.append(parent1.item_ids[i])
                child2_items.append(parent2.item_ids[i])
            else:
                child1_items.append(parent2.item_ids[i])
                child2_items.append(parent1.item_ids[i])

        # Remove duplicates and fill with random items
        child1_items = list(dict.fromkeys(child1_items))
        child2_items = list(dict.fromkeys(child2_items))

        return Individual(item_ids=child1_items), Individual(item_ids=child2_items)

    def mutate(self, individual: Individual, candidate_items: List[str]) -> Individual:
        """
        Mutation: randomly replace items in recommendation list.
        """
        mutated_items = individual.item_ids.copy()

        for i in range(len(mutated_items)):
            if random.random() < self.mutation_rate:
                # Replace with random item not in current list
                available = [item for item in candidate_items if item not in mutated_items]
                if available:
                    mutated_items[i] = random.choice(available)

        return Individual(item_ids=mutated_items)

    def environmental_selection(self, combined: List[Individual]) -> List[Individual]:
        """
        Environmental selection using NSGA-II style.
        """
        fronts = self.non_dominated_sort(combined)

        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= self.population_size:
                new_population.extend(front)
            else:
                # Need to truncate this front
                self.calculate_crowding_distance(front)
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                remaining = self.population_size - len(new_population)
                new_population.extend(front[:remaining])
                break

        return new_population

    def update_archive(self, population: List[Individual]) -> None:
        """Update Pareto archive with non-dominated feasible solutions."""
        feasible = [ind for ind in population if ind.is_feasible]
        if not feasible:
            # If no feasible solutions, keep best infeasible
            feasible = sorted(population, key=lambda x: x.total_violation)[:10]

        combined = self.archive + feasible
        fronts = self.non_dominated_sort(combined)

        if fronts:
            self.archive = fronts[0][:self.population_size]

    def get_best_solution(self) -> Optional[Individual]:
        """Get best solution from archive or population."""
        if self.archive:
            # Return solution with best average normalized score
            return max(self.archive, key=lambda x: np.mean(x.scores) if x.is_feasible else -x.total_violation)
        if self.population:
            feasible = [ind for ind in self.population if ind.is_feasible]
            if feasible:
                return max(feasible, key=lambda x: np.mean(x.scores))
            return min(self.population, key=lambda x: x.total_violation)
        return None

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if not self.population:
            return {}

        feasible = [ind for ind in self.population if ind.is_feasible]
        feasibility_rate = len(feasible) / len(self.population)

        avg_scores = np.mean([ind.scores for ind in self.population], axis=0)
        avg_violations = np.mean([ind.total_violation for ind in self.population])

        return {
            'feasibility_rate': feasibility_rate,
            'avg_objective_1': avg_scores[0],
            'avg_objective_2': avg_scores[1],
            'avg_objective_3': avg_scores[2],
            'avg_violation': avg_violations,
            'archive_size': len(self.archive),
        }
