"""
LLM Coordinator for DualAgent-Rec.
Coordinates resource allocation between Exploitation and Exploration agents.
"""

import sys
import os

# Add shared module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'shared'))

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging

try:
    from llm_utils import OllamaLLM
except ImportError:
    # Fallback for direct imports
    from shared.llm_utils import OllamaLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CoordinatorConfig:
    """Configuration for LLM Coordinator."""
    model_name: str = 'qwen2.5:72b'
    temperature: float = 0.1
    update_frequency: int = 10  # Update every N generations
    use_llm: bool = True  # Can disable LLM for ablation study


class LLMCoordinator:
    """
    LLM-based coordinator for dual-agent resource allocation.

    Responsibilities:
    1. Analyze agent performance metrics
    2. Determine resource allocation ratio
    3. Provide constraint adjustment suggestions
    """

    SYSTEM_PROMPT = """You are an intelligent coordinator for a multi-objective recommendation system.
Your role is to analyze the current state of two optimization agents and determine the optimal resource allocation.

Exploitation Agent: Focuses on accuracy and relevance (CTR prediction)
Exploration Agent: Focuses on diversity and novelty

Key principles:
1. Early optimization should favor exploration (ratio < 0.5 for exploitation)
2. As optimization progresses, gradually shift to exploitation (ratio > 0.5)
3. High constraint violations → increase exploration to find feasible regions
4. Good accuracy but low diversity → increase exploration
5. Good diversity but low accuracy → increase exploitation

Always respond with a JSON object containing:
{
    "exploitation_ratio": <float 0.0-1.0>,
    "reasoning": "<brief explanation>"
}"""

    def __init__(self, config: Optional[CoordinatorConfig] = None):
        """
        Initialize LLM Coordinator.

        Args:
            config: Coordinator configuration
        """
        self.config = config or CoordinatorConfig()
        self.llm = None
        self.generation = 0
        self.allocation_history: List[float] = []

        if self.config.use_llm:
            try:
                self.llm = OllamaLLM(
                    model=self.config.model_name,
                    temperature=self.config.temperature,
                    max_tokens=256
                )
                logger.info(f"LLM Coordinator initialized with {self.config.model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}. Using heuristic coordinator.")
                self.config.use_llm = False

    def get_resource_allocation(
        self,
        exploitation_metrics: Dict[str, float],
        exploration_metrics: Dict[str, float],
        constraint_metrics: Dict[str, float],
        current_generation: int,
        max_generations: int,
        user_profile: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, str]:
        """
        Determine resource allocation between agents.

        Args:
            exploitation_metrics: Performance metrics from exploitation agent
            exploration_metrics: Performance metrics from exploration agent
            constraint_metrics: Current constraint satisfaction metrics
            current_generation: Current generation number
            max_generations: Maximum generations
            user_profile: Optional user profile information

        Returns:
            Tuple of (exploitation_ratio, reasoning)
        """
        self.generation = current_generation

        # Check if should update (based on frequency)
        if current_generation % self.config.update_frequency != 0 and self.allocation_history:
            return self.allocation_history[-1], "Using cached allocation"

        if self.config.use_llm and self.llm:
            ratio, reasoning = self._llm_allocation(
                exploitation_metrics,
                exploration_metrics,
                constraint_metrics,
                current_generation,
                max_generations,
                user_profile
            )
        else:
            ratio, reasoning = self._heuristic_allocation(
                exploitation_metrics,
                exploration_metrics,
                constraint_metrics,
                current_generation,
                max_generations
            )

        self.allocation_history.append(ratio)
        return ratio, reasoning

    def _llm_allocation(
        self,
        exploitation_metrics: Dict[str, float],
        exploration_metrics: Dict[str, float],
        constraint_metrics: Dict[str, float],
        current_generation: int,
        max_generations: int,
        user_profile: Optional[Dict[str, Any]]
    ) -> Tuple[float, str]:
        """Use LLM to determine allocation."""
        prompt = f"""Current optimization state:

Progress: Generation {current_generation} / {max_generations} ({100*current_generation/max_generations:.1f}%)

Exploitation Agent Performance:
- Accuracy (NDCG): {exploitation_metrics.get('ndcg', 0):.4f}
- Hit Rate: {exploitation_metrics.get('hr', 0):.4f}
- Feasibility Rate: {exploitation_metrics.get('feasibility_rate', 0):.2%}

Exploration Agent Performance:
- Diversity Score: {exploration_metrics.get('diversity', 0):.4f}
- Coverage: {exploration_metrics.get('coverage', 0):.4f}
- Decision Space Diversity: {exploration_metrics.get('decision_space_diversity', 0):.4f}

Constraint Satisfaction:
- Overall Feasibility: {constraint_metrics.get('overall_feasibility', 0):.2%}
- Fairness Violation: {constraint_metrics.get('fairness_violation', 0):.4f}
- Seller Coverage Violation: {constraint_metrics.get('seller_violation', 0):.4f}
- New Item Violation: {constraint_metrics.get('new_item_violation', 0):.4f}

{f"User Profile: {user_profile}" if user_profile else ""}

What should be the resource allocation ratio for the Exploitation Agent?
Respond with JSON: {{"exploitation_ratio": <0.0-1.0>, "reasoning": "<explanation>"}}"""

        try:
            response = self.llm.generate(prompt, self.SYSTEM_PROMPT)

            # Parse JSON response
            import json
            # Try to extract JSON from response
            if '{' in response:
                json_str = response[response.index('{'):response.rindex('}')+1]
                result = json.loads(json_str)
                ratio = float(result.get('exploitation_ratio', 0.5))
                reasoning = result.get('reasoning', 'LLM decision')
                return max(0.0, min(1.0, ratio)), reasoning
        except Exception as e:
            logger.warning(f"LLM allocation failed: {e}")

        # Fallback to heuristic
        return self._heuristic_allocation(
            exploitation_metrics,
            exploration_metrics,
            constraint_metrics,
            current_generation,
            max_generations
        )

    def _heuristic_allocation(
        self,
        exploitation_metrics: Dict[str, float],
        exploration_metrics: Dict[str, float],
        constraint_metrics: Dict[str, float],
        current_generation: int,
        max_generations: int
    ) -> Tuple[float, str]:
        """
        Heuristic-based resource allocation.

        Based on entropy-dominance adaptive mechanism from ARMTCMO.
        """
        progress = current_generation / max_generations

        # Base ratio: linear increase from exploration to exploitation
        base_ratio = 0.3 + 0.5 * progress

        # Adjust based on constraint satisfaction
        feasibility = constraint_metrics.get('overall_feasibility', 0.5)
        if feasibility < 0.3:
            # Too many infeasible solutions, explore more
            base_ratio -= 0.2
        elif feasibility > 0.8:
            # Most solutions feasible, can exploit more
            base_ratio += 0.1

        # Adjust based on diversity
        diversity = exploration_metrics.get('decision_space_diversity', 0.5)
        if diversity < 0.3:
            # Low diversity, explore more
            base_ratio -= 0.15
        elif diversity > 0.7:
            # High diversity, can exploit more
            base_ratio += 0.1

        # Adjust based on accuracy
        accuracy = exploitation_metrics.get('ndcg', 0)
        if accuracy > 0.7:
            # Good accuracy, can explore more
            base_ratio -= 0.1
        elif accuracy < 0.3:
            # Poor accuracy, exploit more
            base_ratio += 0.1

        ratio = max(0.2, min(0.8, base_ratio))

        reasoning = f"Heuristic: progress={progress:.2f}, feasibility={feasibility:.2f}, diversity={diversity:.2f}"
        return ratio, reasoning

    def get_constraint_suggestions(
        self,
        constraint_metrics: Dict[str, float],
        history_length: int = 10
    ) -> Dict[str, float]:
        """
        Get suggestions for constraint threshold adjustments.

        Returns:
            Dict of adjustment factors for each constraint
        """
        suggestions = {
            'fairness_adjustment': 0.0,
            'seller_coverage_adjustment': 0.0,
            'new_item_adjustment': 0.0,
        }

        # Based on violation history, suggest relaxation or tightening
        if constraint_metrics.get('fairness_violation', 0) > 0.3:
            suggestions['fairness_adjustment'] = -0.05  # Relax
        elif constraint_metrics.get('fairness_violation', 0) < 0.05:
            suggestions['fairness_adjustment'] = 0.02  # Tighten

        if constraint_metrics.get('seller_violation', 0) > 0.3:
            suggestions['seller_coverage_adjustment'] = -0.05
        elif constraint_metrics.get('seller_violation', 0) < 0.05:
            suggestions['seller_coverage_adjustment'] = 0.02

        if constraint_metrics.get('new_item_violation', 0) > 0.3:
            suggestions['new_item_adjustment'] = -0.05
        elif constraint_metrics.get('new_item_violation', 0) < 0.05:
            suggestions['new_item_adjustment'] = 0.02

        return suggestions

    def get_coordination_summary(self) -> Dict[str, Any]:
        """Get summary of coordination decisions."""
        if not self.allocation_history:
            return {}

        return {
            'total_decisions': len(self.allocation_history),
            'avg_exploitation_ratio': np.mean(self.allocation_history),
            'ratio_std': np.std(self.allocation_history),
            'ratio_trend': (
                'increasing' if len(self.allocation_history) > 5 and
                np.mean(self.allocation_history[-5:]) > np.mean(self.allocation_history[:5])
                else 'stable_or_decreasing'
            ),
            'final_ratio': self.allocation_history[-1] if self.allocation_history else 0.5,
        }
