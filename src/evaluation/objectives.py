"""
Objectives Calculator for DualAgent-Rec.
Calculates CTR, diversity, and novelty objectives.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Set
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity


class ObjectivesCalculator:
    """
    Calculate multi-objective scores for recommendation lists.

    Objectives:
    - f1: Relevance/CTR prediction
    - f2: Intra-list Diversity
    - f3: Novelty (inverse popularity)
    """

    def __init__(
        self,
        item_embeddings: Optional[Dict[str, np.ndarray]] = None,
        item_popularity: Optional[Dict[str, float]] = None
    ):
        """
        Initialize objectives calculator.

        Args:
            item_embeddings: Item ID -> embedding vector mapping
            item_popularity: Item ID -> popularity score mapping
        """
        self.item_embeddings = item_embeddings or {}
        self.item_popularity = item_popularity or {}

    def calculate(
        self,
        recommended_items: List[str],
        user_history: List[Dict],
        item_features: Dict[str, Dict]
    ) -> List[float]:
        """
        Calculate all objective scores.

        Returns:
            [relevance_score, diversity_score, novelty_score]
        """
        relevance = self._calculate_relevance(recommended_items, user_history, item_features)
        diversity = self._calculate_diversity(recommended_items, item_features)
        novelty = self._calculate_novelty(recommended_items, item_features)

        return [relevance, diversity, novelty]

    def _calculate_relevance(
        self,
        items: List[str],
        user_history: List[Dict],
        item_features: Dict[str, Dict]
    ) -> float:
        """
        Calculate relevance/CTR prediction score.

        Uses category overlap and embedding similarity with user history.
        """
        if not items or not user_history:
            return 0.0

        # Get user's preferred categories (support both 'category' and 'main_category')
        user_categories = Counter([
            h.get('category') or h.get('main_category', 'Unknown') for h in user_history
        ])
        total_interactions = sum(user_categories.values())

        relevance_scores = []

        for item_id in items:
            item_info = item_features.get(item_id, {})
            item_category = item_info.get('category') or item_info.get('main_category', 'Unknown')

            # Category match score
            category_weight = user_categories.get(item_category, 0) / total_interactions

            # Embedding similarity (if available)
            embedding_sim = 0.0
            if item_id in self.item_embeddings:
                item_emb = self.item_embeddings[item_id]
                user_emb = self._get_user_embedding(user_history)
                if user_emb is not None:
                    embedding_sim = float(cosine_similarity(
                        item_emb.reshape(1, -1),
                        user_emb.reshape(1, -1)
                    )[0, 0])

            # Combined score
            score = 0.6 * category_weight + 0.4 * max(0, embedding_sim)
            relevance_scores.append(score)

        # DCG-weighted average (higher position = more weight)
        k = len(relevance_scores)
        weights = 1 / np.log2(np.arange(2, k + 2))
        weighted_score = np.sum(np.array(relevance_scores) * weights) / np.sum(weights)

        return float(weighted_score)

    def _get_user_embedding(self, user_history: List[Dict]) -> Optional[np.ndarray]:
        """Get aggregated user embedding from history."""
        embeddings = []
        for h in user_history[-20:]:  # Last 20 items
            item_id = h.get('item_id', '')
            if item_id in self.item_embeddings:
                embeddings.append(self.item_embeddings[item_id])

        if embeddings:
            return np.mean(embeddings, axis=0)
        return None

    def _calculate_diversity(
        self,
        items: List[str],
        item_features: Dict[str, Dict]
    ) -> float:
        """
        Calculate Intra-List Diversity (ILD).

        ILD = (2 / k(k-1)) * Σ distance(i, j) for all pairs
        """
        if len(items) < 2:
            return 0.0

        k = len(items)
        total_distance = 0.0
        pair_count = 0

        for i in range(k):
            for j in range(i + 1, k):
                distance = self._item_distance(items[i], items[j], item_features)
                total_distance += distance
                pair_count += 1

        ild = total_distance / pair_count if pair_count > 0 else 0.0
        return float(ild)

    def _item_distance(
        self,
        item1: str,
        item2: str,
        item_features: Dict[str, Dict]
    ) -> float:
        """Calculate distance between two items."""
        # Try embedding distance first
        if item1 in self.item_embeddings and item2 in self.item_embeddings:
            similarity = cosine_similarity(
                self.item_embeddings[item1].reshape(1, -1),
                self.item_embeddings[item2].reshape(1, -1)
            )[0, 0]
            return 1 - similarity

        # Fall back to category-based distance
        item1_info = item_features.get(item1, {})
        item2_info = item_features.get(item2, {})
        cat1 = item1_info.get('category') or item1_info.get('main_category', 'Unknown')
        cat2 = item2_info.get('category') or item2_info.get('main_category', 'Unknown')

        if cat1 == cat2:
            return 0.2  # Same category = low diversity
        else:
            return 0.8  # Different category = high diversity

    def _calculate_novelty(
        self,
        items: List[str],
        item_features: Dict[str, Dict]
    ) -> float:
        """
        Calculate novelty score (inverse popularity).

        Novelty = avg(1 - popularity(item)) for all items
        """
        if not items:
            return 0.0

        novelty_scores = []

        for item_id in items:
            # Get popularity (0 = unpopular/novel, 1 = very popular)
            if item_id in self.item_popularity:
                popularity = self.item_popularity[item_id]
            else:
                # Default to medium popularity
                popularity = item_features.get(item_id, {}).get('popularity', 0.5)

            novelty = 1 - popularity
            novelty_scores.append(novelty)

        return float(np.mean(novelty_scores))

    def update_popularity(self, interaction_counts: Dict[str, int]) -> None:
        """Update item popularity from interaction counts."""
        if not interaction_counts:
            return

        max_count = max(interaction_counts.values())
        for item_id, count in interaction_counts.items():
            self.item_popularity[item_id] = count / max_count


class RecommendationMetrics:
    """
    Standard recommendation evaluation metrics.
    """

    @staticmethod
    def ndcg_at_k(recommended: List[str], relevant: Set[str], k: int = 10) -> float:
        """
        Calculate NDCG@k.
        """
        recommended = recommended[:k]
        dcg = 0.0
        for i, item in enumerate(recommended):
            if item in relevant:
                dcg += 1 / np.log2(i + 2)

        # Ideal DCG
        ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))

        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    @staticmethod
    def hit_rate_at_k(recommended: List[str], relevant: Set[str], k: int = 10) -> float:
        """
        Calculate Hit Rate@k.
        """
        recommended = recommended[:k]
        hits = len(set(recommended) & relevant)
        return 1.0 if hits > 0 else 0.0

    @staticmethod
    def mrr(recommended: List[str], relevant: Set[str]) -> float:
        """
        Calculate Mean Reciprocal Rank.
        """
        for i, item in enumerate(recommended):
            if item in relevant:
                return 1 / (i + 1)
        return 0.0

    @staticmethod
    def coverage(recommended_all: List[List[str]], all_items: Set[str]) -> float:
        """
        Calculate catalog coverage.
        """
        recommended_items = set()
        for rec_list in recommended_all:
            recommended_items.update(rec_list)
        return len(recommended_items) / len(all_items) if all_items else 0.0


class MultiObjectiveMetrics:
    """
    Multi-objective optimization evaluation metrics.
    """

    @staticmethod
    def hypervolume(pareto_front: List[np.ndarray], reference_point: np.ndarray) -> float:
        """
        Calculate hypervolume indicator.

        Simple 2D/3D implementation using Monte Carlo estimation.
        """
        if not pareto_front:
            return 0.0

        pareto_front = np.array(pareto_front)
        n_samples = 10000
        n_dims = pareto_front.shape[1]

        # Generate random points in the bounded region
        random_points = np.random.uniform(
            low=np.zeros(n_dims),
            high=reference_point,
            size=(n_samples, n_dims)
        )

        # Count points dominated by the Pareto front
        dominated_count = 0
        for point in random_points:
            for pf_point in pareto_front:
                if np.all(pf_point >= point):
                    dominated_count += 1
                    break

        # Estimate hypervolume
        volume_of_region = np.prod(reference_point)
        hv = (dominated_count / n_samples) * volume_of_region

        return float(hv)

    @staticmethod
    def igd(pareto_front: List[np.ndarray], true_front: List[np.ndarray]) -> float:
        """
        Calculate Inverted Generational Distance (IGD).

        Lower is better.
        """
        if not pareto_front or not true_front:
            return float('inf')

        pareto_front = np.array(pareto_front)
        true_front = np.array(true_front)

        distances = []
        for true_point in true_front:
            min_dist = min(np.linalg.norm(true_point - pf_point) for pf_point in pareto_front)
            distances.append(min_dist)

        return float(np.mean(distances))

    @staticmethod
    def spacing(pareto_front: List[np.ndarray]) -> float:
        """
        Calculate spacing metric (solution distribution uniformity).

        Lower is better (more uniform).
        """
        if len(pareto_front) < 2:
            return 0.0

        pareto_front = np.array(pareto_front)
        n = len(pareto_front)

        # Calculate minimum distances
        min_distances = []
        for i in range(n):
            distances = [
                np.linalg.norm(pareto_front[i] - pareto_front[j])
                for j in range(n) if i != j
            ]
            min_distances.append(min(distances))

        d_mean = np.mean(min_distances)
        spacing = np.sqrt(np.sum((np.array(min_distances) - d_mean) ** 2) / n)

        return float(spacing)
