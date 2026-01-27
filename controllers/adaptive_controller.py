from typing import List, Dict, Optional, Tuple
from estimators.difficulty_estimator import DifficultyEstimator
import torch

class AdaptiveController:
    """
    Adaptive controller that uses difficulty estimator to filter variants
    and allocate compute intelligently
    """
    
    def __init__(
        self,
        estimator: DifficultyEstimator,
        min_solve_prob: float = 0.1,
        max_verifier_cost: float = 5.0,
        min_curriculum_value: float = 0.3,
        max_variants_per_problem: int = 50,
        compute_budget_per_problem: float = 100.0
    ):
        """
        Initialize adaptive controller
        
        Args:
            estimator: Trained difficulty estimator
            min_solve_prob: Minimum solve probability to accept variant
            max_verifier_cost: Maximum verification cost (seconds)
            min_curriculum_value: Minimum curriculum value
            max_variants_per_problem: Max variants to generate per problem
            compute_budget_per_problem: Total compute budget (seconds)
        """
        self.estimator = estimator
        self.min_solve_prob = min_solve_prob
        self.max_verifier_cost = max_verifier_cost
        self.min_curriculum_value = min_curriculum_value
        self.max_variants_per_problem = max_variants_per_problem
        self.compute_budget = compute_budget_per_problem
        
        self.stats = {
            'variants_generated': 0,
            'variants_accepted': 0,
            'variants_rejected': 0,
            'rejection_reasons': {'too_hard': 0, 'too_expensive': 0, 'low_value': 0},
            'compute_used': 0.0,
            'compute_saved': 0.0
        }
    
    def should_accept_variant(
        self,
        variant_text: str,
        depth: int,
        similarity_to_root: float,
        root_problem: str
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Decide if variant should be accepted
        
        Returns:
            Tuple of (should_accept, prediction_dict)
        """
        variant_length = len(variant_text)
        
        # Get predictions from estimator
        predictions = self.estimator.predict(
            variant_text=variant_text,
            depth=depth,
            similarity=similarity_to_root,
            variant_length=variant_length
        )
        
        solve_prob = predictions['solve_prob']
        verifier_cost = predictions['verifier_cost']
        curriculum_value = predictions['curriculum_value']
        
        # Check budget
        if self.stats['compute_used'] + verifier_cost > self.compute_budget:
            self.stats['variants_rejected'] += 1
            self.stats['rejection_reasons']['too_expensive'] += 1
            return False, predictions
        
        # Check limits
        if self.stats['variants_accepted'] >= self.max_variants_per_problem:
            self.stats['variants_rejected'] += 1
            return False, predictions
        
        # Adaptive filtering
        reject_reason = None
        
        if solve_prob < self.min_solve_prob:
            reject_reason = 'too_hard'
        elif verifier_cost > self.max_verifier_cost:
            reject_reason = 'too_expensive'
        elif curriculum_value < self.min_curriculum_value:
            reject_reason = 'low_value'
        
        if reject_reason:
            self.stats['variants_rejected'] += 1
            self.stats['rejection_reasons'][reject_reason] += 1
            self.stats['compute_saved'] += verifier_cost
            return False, predictions
        
        # Accept variant
        self.stats['variants_accepted'] += 1
        self.stats['compute_used'] += verifier_cost
        return True, predictions
    
    def reset_stats(self):
        """Reset statistics for new problem"""
        self.stats = {
            'variants_generated': 0,
            'variants_accepted': 0,
            'variants_rejected': 0,
            'rejection_reasons': {'too_hard': 0, 'too_expensive': 0, 'low_value': 0},
            'compute_used': 0.0,
            'compute_saved': 0.0
        }
    
    def get_efficiency_gain(self) -> float:
        """Calculate efficiency gain (compute saved / compute used)"""
        total = self.stats['compute_used'] + self.stats['compute_saved']
        if total == 0:
            return 0.0
        return self.stats['compute_saved'] / total
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        return self.stats.copy()
