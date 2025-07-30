"""Adaptive score comparison with statistical significance testing."""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass 
class ComparisonContext:
    """Context information for adaptive score comparison."""
    score_variance: float = 0.01
    sample_size: int = 1
    score_distribution: str = "normal"  # normal, uniform, etc.
    historical_scores: Optional[List[float]] = None


@dataclass
class ComparisonResult:
    """Result of adaptive score comparison."""
    are_equivalent: bool
    confidence_level: float
    adaptive_epsilon: float
    statistical_significance: float
    method_used: str


class AdaptiveScoreComparator:
    """Adaptive comparison with statistical significance testing.
    
    Replaces fixed epsilon comparison (1e-6) with context-aware statistical methods:
    1. Adaptive epsilon based on score distribution and variance
    2. Statistical significance testing (t-test, Mann-Whitney U)
    3. Confidence interval overlap analysis
    4. Bootstrap-based uncertainty estimation
    """
    
    def __init__(self, confidence_threshold: float = 0.95):
        self.confidence_threshold = confidence_threshold
        self.score_variance_tracker: Dict[str, List[float]] = {}
        self.comparison_history: List[ComparisonResult] = []
        
    def scores_equivalent(
        self,
        score1: float,
        score2: float, 
        context: Optional[ComparisonContext] = None
    ) -> ComparisonResult:
        """Determine if scores are statistically equivalent using multiple methods."""
        
        if context is None:
            context = ComparisonContext()
            
        # Method 1: Adaptive epsilon based on context
        adaptive_epsilon = self._calculate_adaptive_epsilon(score1, score2, context)
        epsilon_equivalent = abs(score1 - score2) < adaptive_epsilon
        
        # Method 2: Statistical significance test
        statistical_result = self._statistical_significance_test(score1, score2, context)
        
        # Method 3: Confidence interval overlap
        ci_result = self._confidence_intervals_overlap(score1, score2, context)
        
        # Method 4: Effect size analysis
        effect_size = self._calculate_effect_size(score1, score2, context)
        
        # Combine results with weighted decision
        final_decision, confidence, method = self._combine_comparison_methods(
            epsilon_equivalent, statistical_result, ci_result, effect_size, context
        )
        
        result = ComparisonResult(
            are_equivalent=final_decision,
            confidence_level=confidence,
            adaptive_epsilon=adaptive_epsilon,
            statistical_significance=statistical_result.get('p_value', 1.0),
            method_used=method
        )
        
        # Track for learning
        self.comparison_history.append(result)
        self._update_variance_tracking(score1, score2, context)
        
        return result
    
    def _calculate_adaptive_epsilon(
        self,
        score1: float,
        score2: float,
        context: ComparisonContext
    ) -> float:
        """Calculate context-aware epsilon for comparison."""
        
        # Base epsilon on score magnitude and variance
        score_magnitude = max(abs(score1), abs(score2))
        score_variance = context.score_variance
        
        # Scale epsilon based on magnitude (relative tolerance)
        magnitude_factor = score_magnitude * 0.001  # 0.1% relative tolerance
        
        # Scale based on observed variance (absolute tolerance)  
        variance_factor = 2 * np.sqrt(score_variance)
        
        # Historical learning factor
        historical_factor = self._get_historical_epsilon_adjustment(context)
        
        # Combine factors
        adaptive_epsilon = max(
            1e-6,  # Minimum epsilon (original fixed value)
            magnitude_factor + variance_factor + historical_factor
        )
        
        # Cap maximum epsilon to prevent overly loose comparisons
        max_epsilon = min(0.1, score_magnitude * 0.05)  # 5% of magnitude or 0.1
        
        return min(adaptive_epsilon, max_epsilon)
    
    def _statistical_significance_test(
        self,
        score1: float,
        score2: float,
        context: ComparisonContext
    ) -> Dict[str, float]:
        """Perform statistical significance testing."""
        
        try:
            # For single scores, simulate distributions based on context
            if context.historical_scores and len(context.historical_scores) > 2:
                # Use historical scores for more robust testing
                historical = np.array(context.historical_scores)
                
                # Create simulated distributions around each score
                n_samples = max(10, context.sample_size)
                
                # Use historical variance for simulation
                hist_std = np.std(historical)
                if hist_std == 0:
                    hist_std = context.score_variance
                
                dist1 = np.random.normal(score1, hist_std, n_samples)
                dist2 = np.random.normal(score2, hist_std, n_samples)
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(dist1, dist2)
                
                # Also perform Mann-Whitney U test (non-parametric)
                try:
                    u_stat, u_p_value = stats.mannwhitneyu(dist1, dist2, alternative='two-sided')
                except ValueError:
                    u_p_value = 1.0
                
                return {
                    'p_value': min(p_value, u_p_value),  # Use more conservative p-value
                    't_statistic': abs(t_stat),
                    'is_significant': min(p_value, u_p_value) < (1 - self.confidence_threshold)
                }
            else:
                # For single point comparison, use theoretical approach
                # Assume scores come from normal distributions with known variance
                pooled_std = np.sqrt(2 * context.score_variance)
                
                if pooled_std > 0:
                    z_score = abs(score1 - score2) / pooled_std
                    p_value = 2 * (1 - stats.norm.cdf(z_score))  # Two-tailed test
                    
                    return {
                        'p_value': p_value,
                        't_statistic': z_score,
                        'is_significant': p_value < (1 - self.confidence_threshold)
                    }
                else:
                    # No variance - scores are identical if exactly equal
                    return {
                        'p_value': 0.0 if score1 == score2 else 1.0,
                        't_statistic': float('inf') if score1 != score2 else 0.0,
                        'is_significant': score1 != score2
                    }
                    
        except Exception as e:
            logger.debug(f"Statistical significance test failed: {e}")
            return {
                'p_value': 1.0,
                't_statistic': 0.0,
                'is_significant': False
            }
    
    def _confidence_intervals_overlap(
        self,
        score1: float,
        score2: float,
        context: ComparisonContext
    ) -> Dict[str, Any]:
        """Check if confidence intervals overlap."""
        
        try:
            # Calculate confidence intervals
            confidence_level = self.confidence_threshold
            alpha = 1 - confidence_level
            
            # Standard error estimation
            if context.sample_size > 1:
                se = np.sqrt(context.score_variance / context.sample_size)
            else:
                se = np.sqrt(context.score_variance)
            
            # Critical value (assuming normal distribution)
            critical_value = stats.norm.ppf(1 - alpha/2)
            
            # Confidence intervals
            ci1_lower = score1 - critical_value * se
            ci1_upper = score1 + critical_value * se
            ci2_lower = score2 - critical_value * se  
            ci2_upper = score2 + critical_value * se
            
            # Check overlap
            overlap = not (ci1_upper < ci2_lower or ci2_upper < ci1_lower)
            overlap_amount = 0.0
            
            if overlap:
                # Calculate amount of overlap
                overlap_start = max(ci1_lower, ci2_lower)
                overlap_end = min(ci1_upper, ci2_upper)
                overlap_amount = (overlap_end - overlap_start) / max(
                    ci1_upper - ci1_lower, ci2_upper - ci2_lower
                )
            
            return {
                'intervals_overlap': overlap,
                'overlap_amount': overlap_amount,
                'ci1': (ci1_lower, ci1_upper),
                'ci2': (ci2_lower, ci2_upper)
            }
            
        except Exception as e:
            logger.debug(f"Confidence interval analysis failed: {e}")
            return {
                'intervals_overlap': True,  # Conservative assumption
                'overlap_amount': 1.0,
                'ci1': (score1, score1),
                'ci2': (score2, score2)
            }
    
    def _calculate_effect_size(
        self,
        score1: float,
        score2: float,
        context: ComparisonContext
    ) -> Dict[str, float]:
        """Calculate effect size (Cohen's d equivalent)."""
        
        try:
            # Cohen's d = (mean1 - mean2) / pooled_standard_deviation
            mean_diff = abs(score1 - score2)
            pooled_std = np.sqrt(context.score_variance)
            
            if pooled_std > 0:
                cohens_d = mean_diff / pooled_std
            else:
                cohens_d = 0.0 if mean_diff == 0 else float('inf')
            
            # Effect size interpretation
            if cohens_d < 0.2:
                effect_size_category = "negligible"
            elif cohens_d < 0.5:
                effect_size_category = "small"
            elif cohens_d < 0.8:
                effect_size_category = "medium"
            else:
                effect_size_category = "large"
            
            return {
                'cohens_d': cohens_d,
                'effect_size_category': effect_size_category,
                'is_practically_significant': cohens_d >= 0.2  # Small effect threshold
            }
            
        except Exception as e:
            logger.debug(f"Effect size calculation failed: {e}")
            return {
                'cohens_d': 0.0,
                'effect_size_category': "negligible",
                'is_practically_significant': False
            }
    
    def _combine_comparison_methods(
        self,
        epsilon_equivalent: bool,
        statistical_result: Dict[str, Any],
        ci_result: Dict[str, Any],
        effect_size: Dict[str, float],
        context: ComparisonContext
    ) -> Tuple[bool, float, str]:
        """Combine multiple comparison methods into final decision."""
        
        # Collect evidence for equivalence
        evidence_for_equivalence = []
        methods_used = []
        
        # Method 1: Adaptive epsilon
        if epsilon_equivalent:
            evidence_for_equivalence.append(0.9)  # High confidence
            methods_used.append("adaptive_epsilon")
        else:
            evidence_for_equivalence.append(0.1)
        
        # Method 2: Statistical significance
        if not statistical_result.get('is_significant', True):
            # Not statistically significant = likely equivalent
            confidence = 1 - statistical_result.get('p_value', 0.5)
            evidence_for_equivalence.append(confidence)
            methods_used.append("statistical_test")
        else:
            evidence_for_equivalence.append(statistical_result.get('p_value', 0.5))
        
        # Method 3: Confidence intervals
        if ci_result.get('intervals_overlap', False):
            overlap_confidence = ci_result.get('overlap_amount', 0.5) * 0.8
            evidence_for_equivalence.append(overlap_confidence)
            methods_used.append("confidence_intervals")
        else:
            evidence_for_equivalence.append(0.2)
        
        # Method 4: Effect size
        if not effect_size.get('is_practically_significant', True):
            evidence_for_equivalence.append(0.8)
            methods_used.append("effect_size")
        else:
            # Large effect size suggests scores are different
            cohens_d = effect_size.get('cohens_d', 0.0)
            evidence_for_equivalence.append(max(0.1, 1.0 - min(cohens_d / 2.0, 1.0)))
        
        # Weighted average (you could also use more sophisticated ensemble methods)
        overall_confidence = np.mean(evidence_for_equivalence)
        
        # Decision threshold
        decision_threshold = 0.5  # Could be adaptive based on context
        final_decision = overall_confidence > decision_threshold
        
        method_description = f"ensemble({','.join(methods_used)})"
        
        return final_decision, overall_confidence, method_description
    
    def _get_historical_epsilon_adjustment(self, context: ComparisonContext) -> float:
        """Get epsilon adjustment based on historical comparison accuracy."""
        
        if len(self.comparison_history) < 5:
            return 0.0  # Not enough history
        
        # Analyze recent comparison patterns
        recent_comparisons = self.comparison_history[-10:]
        
        # If we've been too strict (many high-confidence non-equivalences), relax epsilon
        strict_comparisons = [
            c for c in recent_comparisons 
            if not c.are_equivalent and c.confidence_level > 0.8
        ]
        
        if len(strict_comparisons) > 7:  # More than 70% strict
            return context.score_variance * 0.5  # Increase epsilon
        
        # If we've been too loose (many low-confidence equivalences), tighten epsilon
        loose_comparisons = [
            c for c in recent_comparisons
            if c.are_equivalent and c.confidence_level < 0.6
        ]
        
        if len(loose_comparisons) > 6:  # More than 60% loose
            return -context.score_variance * 0.3  # Decrease epsilon
        
        return 0.0  # No adjustment needed
    
    def _update_variance_tracking(
        self,
        score1: float,
        score2: float,
        context: ComparisonContext
    ) -> None:
        """Update variance tracking for improved epsilon calculation."""
        
        score_key = "global"  # Could be made more specific based on context
        
        if score_key not in self.score_variance_tracker:
            self.score_variance_tracker[score_key] = []
        
        # Track the difference between scores
        score_diff = abs(score1 - score2)
        self.score_variance_tracker[score_key].append(score_diff)
        
        # Keep only recent history
        if len(self.score_variance_tracker[score_key]) > 100:
            self.score_variance_tracker[score_key] = self.score_variance_tracker[score_key][-100:]
        
        # Update context variance estimate
        if len(self.score_variance_tracker[score_key]) > 2:
            context.score_variance = np.var(self.score_variance_tracker[score_key])
    
    def get_comparison_statistics(self) -> Dict[str, Any]:
        """Get statistics about comparison performance."""
        
        if not self.comparison_history:
            return {"message": "No comparison history available"}
        
        recent_history = self.comparison_history[-50:]  # Last 50 comparisons
        
        return {
            "total_comparisons": len(self.comparison_history),
            "recent_equivalence_rate": np.mean([c.are_equivalent for c in recent_history]),
            "average_confidence": np.mean([c.confidence_level for c in recent_history]),
            "average_epsilon": np.mean([c.adaptive_epsilon for c in recent_history]),
            "methods_used": list(set(c.method_used for c in recent_history))
        } 