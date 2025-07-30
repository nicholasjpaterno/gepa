"""Advanced complementarity analysis replacing simple score difference heuristics."""

from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import stats
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ComplementarityType(Enum):
    """Types of complementarity patterns."""
    STATISTICAL = "statistical"
    PATTERN_BASED = "pattern_based" 
    ENSEMBLE = "ensemble"
    INSTANCE_SPECIFIC = "instance_specific"


@dataclass
class StatisticalAnalysis:
    """Statistical analysis of complementarity."""
    significance_level: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    test_statistic: float
    p_value: float
    test_method: str


@dataclass
class PatternAnalysis:
    """Pattern-based complementarity analysis."""
    strength_score: float
    instance_patterns: Dict[str, List[int]]
    complementary_regions: List[Tuple[float, float]]
    pattern_confidence: float
    pattern_type: str


@dataclass
class EnsembleAnalysis:
    """Ensemble performance analysis."""
    expected_improvement: float
    confidence_interval: Tuple[float, float]
    optimal_weights: Dict[str, float]
    synergy_score: float
    prediction_method: str


@dataclass
class ComplementarityAnalysis:
    """Comprehensive complementarity analysis result."""
    statistical_significance: float
    complementarity_strength: float
    predicted_ensemble_gain: float
    confidence_interval: Tuple[float, float]
    complementary_instance_patterns: Dict[str, List[int]]
    overall_score: float
    analysis_confidence: float
    explanation: str


class StatisticalSignificanceTester:
    """Statistical tests for performance complementarity."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.test_history: List[Dict[str, Any]] = []
        
    def test_complementarity(
        self,
        scores1: Dict[int, float],
        scores2: Dict[int, float]
    ) -> StatisticalAnalysis:
        """Test statistical significance of complementarity."""
        
        try:
            # Get common instances
            common_instances = set(scores1.keys()) & set(scores2.keys())
            if len(common_instances) < 3:
                return self._default_analysis("Insufficient data for statistical testing")
            
            # Extract paired scores
            paired_scores1 = [scores1[i] for i in common_instances]
            paired_scores2 = [scores2[i] for i in common_instances]
            
            # Test 1: Paired t-test for mean differences
            paired_result = self._paired_ttest(paired_scores1, paired_scores2)
            
            # Test 2: Wilcoxon signed-rank test (non-parametric)
            wilcoxon_result = self._wilcoxon_test(paired_scores1, paired_scores2)
            
            # Test 3: Bootstrap confidence intervals
            bootstrap_ci = self._bootstrap_confidence_interval(paired_scores1, paired_scores2)
            
            # Test 4: Effect size analysis
            effect_size = self._calculate_effect_size(paired_scores1, paired_scores2)
            
            # Combine results (use most conservative p-value)
            final_p_value = max(paired_result.get('p_value', 1.0), wilcoxon_result.get('p_value', 1.0))
            significance_level = 1.0 - final_p_value
            
            # Select best test method
            if len(common_instances) >= 30:
                test_method = "paired_t_test"
                test_stat = paired_result.get('t_statistic', 0.0)
            else:
                test_method = "wilcoxon_signed_rank"
                test_stat = wilcoxon_result.get('w_statistic', 0.0)
            
            result = StatisticalAnalysis(
                significance_level=significance_level,
                effect_size=effect_size,
                confidence_interval=bootstrap_ci,
                test_statistic=test_stat,
                p_value=final_p_value,
                test_method=test_method
            )
            
            # Record test for learning
            self.test_history.append({
                'n_instances': len(common_instances),
                'p_value': final_p_value,
                'effect_size': effect_size,
                'method': test_method
            })
            
            return result
            
        except Exception as e:
            logger.debug(f"Statistical testing failed: {e}")
            return self._default_analysis(f"Statistical testing error: {str(e)}")
    
    def _paired_ttest(
        self,
        scores1: List[float],
        scores2: List[float]
    ) -> Dict[str, float]:
        """Perform paired t-test."""
        
        try:
            # Calculate differences
            differences = [s1 - s2 for s1, s2 in zip(scores1, scores2)]
            
            if len(differences) < 2:
                return {'t_statistic': 0.0, 'p_value': 1.0}
            
            # One-sample t-test on differences (H0: mean difference = 0)
            t_stat, p_value = stats.ttest_1samp(differences, 0)
            
            return {
                't_statistic': abs(t_stat),
                'p_value': p_value,
                'degrees_freedom': len(differences) - 1
            }
            
        except Exception as e:
            logger.debug(f"Paired t-test failed: {e}")
            return {'t_statistic': 0.0, 'p_value': 1.0}
    
    def _wilcoxon_test(
        self,
        scores1: List[float],
        scores2: List[float]
    ) -> Dict[str, float]:
        """Perform Wilcoxon signed-rank test."""
        
        try:
            # Calculate differences
            differences = [s1 - s2 for s1, s2 in zip(scores1, scores2)]
            
            # Remove zero differences
            non_zero_diff = [d for d in differences if d != 0]
            
            if len(non_zero_diff) < 3:
                return {'w_statistic': 0.0, 'p_value': 1.0}
            
            # Wilcoxon signed-rank test
            w_stat, p_value = stats.wilcoxon(non_zero_diff)
            
            return {
                'w_statistic': w_stat,
                'p_value': p_value,
                'n_non_zero': len(non_zero_diff)
            }
            
        except Exception as e:
            logger.debug(f"Wilcoxon test failed: {e}")
            return {'w_statistic': 0.0, 'p_value': 1.0}
    
    def _bootstrap_confidence_interval(
        self,
        scores1: List[float],
        scores2: List[float],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for mean difference."""
        
        try:
            differences = [s1 - s2 for s1, s2 in zip(scores1, scores2)]
            
            if len(differences) < 2:
                return (0.0, 0.0)
            
            # Bootstrap resampling
            bootstrap_means = []
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(differences, size=len(differences), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            # Calculate confidence interval
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(bootstrap_means, lower_percentile)
            upper_bound = np.percentile(bootstrap_means, upper_percentile)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            logger.debug(f"Bootstrap CI calculation failed: {e}")
            return (0.0, 0.0)
    
    def _calculate_effect_size(
        self,
        scores1: List[float],
        scores2: List[float]
    ) -> float:
        """Calculate Cohen's d effect size."""
        
        try:
            differences = [s1 - s2 for s1, s2 in zip(scores1, scores2)]
            
            if len(differences) < 2:
                return 0.0
            
            mean_diff = np.mean(differences)  
            std_diff = np.std(differences, ddof=1)
            
            if std_diff == 0:
                return 0.0 if mean_diff == 0 else float('inf')
            
            # Cohen's d for paired samples
            cohens_d = mean_diff / std_diff
            
            return abs(cohens_d)  # Return absolute effect size
            
        except Exception as e:
            logger.debug(f"Effect size calculation failed: {e}")
            return 0.0
    
    def _default_analysis(self, reason: str) -> StatisticalAnalysis:
        """Return default statistical analysis when tests fail."""
        
        return StatisticalAnalysis(
            significance_level=0.5,
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            test_statistic=0.0,
            p_value=1.0,
            test_method=f"default ({reason})"
        )


class PerformancePatternAnalyzer:
    """Analyze complementary performance patterns."""
    
    def __init__(self):
        self.pattern_cache: Dict[str, PatternAnalysis] = {}
        
    def find_complementary_patterns(
        self,
        scores1: Dict[int, float],
        scores2: Dict[int, float],
        instance_metadata: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> PatternAnalysis:
        """Find complementary patterns in performance scores."""
        
        try:
            # Get common instances
            common_instances = list(set(scores1.keys()) & set(scores2.keys()))
            if len(common_instances) < 3:
                return self._default_pattern_analysis("Insufficient data for pattern analysis")
            
            # Pattern 1: Instance-specific complementarity
            instance_patterns = self._find_instance_specific_patterns(
                scores1, scores2, common_instances
            )
            
            # Pattern 2: Performance range complementarity
            range_patterns = self._find_range_complementarity(
                scores1, scores2, common_instances
            )
            
            # Pattern 3: Difficulty-based complementarity
            difficulty_patterns = self._find_difficulty_based_patterns(
                scores1, scores2, common_instances, instance_metadata
            )
            
            # Pattern 4: Error pattern complementarity
            error_patterns = self._find_error_pattern_complementarity(
                scores1, scores2, common_instances
            )
            
            # Combine pattern analyses
            strength_score = self._calculate_pattern_strength(
                instance_patterns, range_patterns, difficulty_patterns, error_patterns
            )
            
            # Identify complementary regions
            complementary_regions = self._identify_complementary_regions(
                scores1, scores2, common_instances
            )
            
            # Calculate confidence
            pattern_confidence = self._calculate_pattern_confidence(
                len(common_instances), instance_patterns, range_patterns
            )
            
            # Determine dominant pattern type
            pattern_type = self._determine_pattern_type(
                instance_patterns, range_patterns, difficulty_patterns, error_patterns
            )
            
            return PatternAnalysis(
                strength_score=strength_score,
                instance_patterns=instance_patterns,
                complementary_regions=complementary_regions,
                pattern_confidence=pattern_confidence,
                pattern_type=pattern_type
            )
            
        except Exception as e:
            logger.debug(f"Pattern analysis failed: {e}")
            return self._default_pattern_analysis(f"Pattern analysis error: {str(e)}")
    
    def _find_instance_specific_patterns(
        self,
        scores1: Dict[int, float],
        scores2: Dict[int, float],
        common_instances: List[int]
    ) -> Dict[str, List[int]]:
        """Find instances where one model significantly outperforms the other."""
        
        patterns = {
            'model1_better': [],  # Instances where model1 >> model2
            'model2_better': [],  # Instances where model2 >> model1
            'both_good': [],      # Instances where both perform well
            'both_poor': [],      # Instances where both perform poorly
            'similar': []         # Instances where performance is similar
        }
        
        for instance_id in common_instances:
            score1 = scores1[instance_id]
            score2 = scores2[instance_id]
            
            diff = score1 - score2
            avg_score = (score1 + score2) / 2
            
            # Threshold for significant difference
            significance_threshold = 0.1
            
            if abs(diff) < significance_threshold:
                patterns['similar'].append(instance_id)
            elif diff > significance_threshold:
                patterns['model1_better'].append(instance_id)
            else:  # diff < -significance_threshold
                patterns['model2_better'].append(instance_id)
            
            # Additional categorization by overall performance
            if avg_score > 0.7:
                patterns['both_good'].append(instance_id)
            elif avg_score < 0.3:
                patterns['both_poor'].append(instance_id)
        
        return patterns
    
    def _find_range_complementarity(
        self,
        scores1: Dict[int, float],
        scores2: Dict[int, float],
        common_instances: List[int]
    ) -> Dict[str, List[int]]:
        """Find complementarity based on performance ranges."""
        
        patterns = {
            'high_performance': [],    # High score range (>0.7)
            'medium_performance': [],  # Medium score range (0.3-0.7)
            'low_performance': []      # Low score range (<0.3)
        }
        
        for instance_id in common_instances:
            score1 = scores1[instance_id]
            score2 = scores2[instance_id]
            
            max_score = max(score1, score2)
            
            if max_score > 0.7:
                patterns['high_performance'].append(instance_id)
            elif max_score > 0.3:
                patterns['medium_performance'].append(instance_id)
            else:
                patterns['low_performance'].append(instance_id)
        
        return patterns
    
    def _find_difficulty_based_patterns(
        self,
        scores1: Dict[int, float],
        scores2: Dict[int, float],
        common_instances: List[int],
        instance_metadata: Optional[Dict[int, Dict[str, Any]]]
    ) -> Dict[str, List[int]]:
        """Find complementarity based on instance difficulty."""
        
        patterns = {
            'easy_instances': [],
            'medium_instances': [],
            'hard_instances': []
        }
        
        if not instance_metadata:
            # Infer difficulty from performance
            for instance_id in common_instances:
                score1 = scores1[instance_id]
                score2 = scores2[instance_id]
                
                avg_score = (score1 + score2) / 2
                
                if avg_score > 0.7:
                    patterns['easy_instances'].append(instance_id)
                elif avg_score > 0.4:
                    patterns['medium_instances'].append(instance_id)
                else:
                    patterns['hard_instances'].append(instance_id)
        else:
            # Use provided difficulty metadata
            for instance_id in common_instances:
                if instance_id in instance_metadata:
                    difficulty = instance_metadata[instance_id].get('difficulty', 'medium')
                    
                    if difficulty in ['easy', 'low']:
                        patterns['easy_instances'].append(instance_id)
                    elif difficulty in ['hard', 'difficult', 'high']:
                        patterns['hard_instances'].append(instance_id)
                    else:
                        patterns['medium_instances'].append(instance_id)
        
        return patterns
    
    def _find_error_pattern_complementarity(
        self,
        scores1: Dict[int, float],
        scores2: Dict[int, float],
        common_instances: List[int]
    ) -> Dict[str, List[int]]:
        """Find complementarity in error patterns."""
        
        patterns = {
            'model1_errors': [],    # Instances where model1 fails but model2 succeeds
            'model2_errors': [],    # Instances where model2 fails but model1 succeeds
            'both_errors': [],      # Instances where both models fail
            'both_success': []      # Instances where both models succeed
        }
        
        error_threshold = 0.5  # Below this is considered failure
        
        for instance_id in common_instances:
            score1 = scores1[instance_id]
            score2 = scores2[instance_id]
            
            model1_success = score1 >= error_threshold
            model2_success = score2 >= error_threshold
            
            if model1_success and model2_success:
                patterns['both_success'].append(instance_id)
            elif model1_success and not model2_success:
                patterns['model2_errors'].append(instance_id)
            elif not model1_success and model2_success:
                patterns['model1_errors'].append(instance_id)
            else:
                patterns['both_errors'].append(instance_id)
        
        return patterns
    
    def _calculate_pattern_strength(
        self,
        instance_patterns: Dict[str, List[int]],
        range_patterns: Dict[str, List[int]],
        difficulty_patterns: Dict[str, List[int]],
        error_patterns: Dict[str, List[int]]
    ) -> float:
        """Calculate overall pattern strength score."""
        
        strength_factors = []
        
        # Factor 1: Instance-specific complementarity
        total_instances = sum(len(instances) for instances in instance_patterns.values())
        if total_instances > 0:
            model1_better = len(instance_patterns.get('model1_better', []))
            model2_better = len(instance_patterns.get('model2_better', []))
            
            # Higher complementarity when both models have instances where they excel
            if model1_better > 0 and model2_better > 0:
                balance = min(model1_better, model2_better) / max(model1_better, model2_better)
                complementarity_factor = (model1_better + model2_better) / total_instances
                instance_strength = balance * complementarity_factor
            else:
                instance_strength = 0.0
            
            strength_factors.append(instance_strength)
        
        # Factor 2: Error pattern complementarity
        model1_errors = len(error_patterns.get('model1_errors', []))
        model2_errors = len(error_patterns.get('model2_errors', []))
        
        if model1_errors > 0 or model2_errors > 0:
            error_complementarity = (model1_errors + model2_errors) / total_instances if total_instances > 0 else 0
            strength_factors.append(error_complementarity)
        
        # Factor 3: Performance range diversity
        range_diversity = len([pattern for pattern, instances in range_patterns.items() if instances])
        range_strength = min(1.0, range_diversity / 3.0)  # 3 ranges maximum
        strength_factors.append(range_strength)
        
        if not strength_factors:
            return 0.0
        
        return np.mean(strength_factors)
    
    def _identify_complementary_regions(
        self,
        scores1: Dict[int, float],
        scores2: Dict[int, float],
        common_instances: List[int],
        n_regions: int = 5
    ) -> List[Tuple[float, float]]:
        """Identify score regions where complementarity is strongest."""
        
        # Divide score space into regions
        score_ranges = np.linspace(0, 1, n_regions + 1)
        complementary_regions = []
        
        for i in range(len(score_ranges) - 1):
            lower_bound = score_ranges[i]
            upper_bound = score_ranges[i + 1]
            
            # Find instances in this score range
            region_instances = []
            for instance_id in common_instances:
                score1 = scores1[instance_id]
                score2 = scores2[instance_id]
                avg_score = (score1 + score2) / 2
                
                if lower_bound <= avg_score < upper_bound:
                    region_instances.append(instance_id)
            
            if len(region_instances) >= 2:
                # Calculate complementarity in this region
                region_complementarity = self._calculate_region_complementarity(
                    scores1, scores2, region_instances
                )
                
                if region_complementarity > 0.3:  # Threshold for significant complementarity
                    complementary_regions.append((lower_bound, upper_bound))
        
        return complementary_regions
    
    def _calculate_region_complementarity(
        self,
        scores1: Dict[int, float],
        scores2: Dict[int, float],
        region_instances: List[int]
    ) -> float:
        """Calculate complementarity within a specific score region."""
        
        differences = []
        for instance_id in region_instances:
            diff = abs(scores1[instance_id] - scores2[instance_id])
            differences.append(diff)
        
        if not differences:
            return 0.0
        
        # Higher differences indicate more complementarity
        return np.mean(differences)
    
    def _calculate_pattern_confidence(
        self,
        n_instances: int,
        instance_patterns: Dict[str, List[int]],
        range_patterns: Dict[str, List[int]]
    ) -> float:
        """Calculate confidence in pattern analysis."""
        
        confidence_factors = []
        
        # Factor 1: Sample size
        if n_instances >= 20:
            size_confidence = 1.0
        elif n_instances >= 10:
            size_confidence = 0.8
        elif n_instances >= 5:
            size_confidence = 0.6
        else:
            size_confidence = 0.3
        
        confidence_factors.append(size_confidence)
        
        # Factor 2: Pattern clarity
        total_instances = sum(len(instances) for instances in instance_patterns.values())
        similar_instances = len(instance_patterns.get('similar', []))
        
        if total_instances > 0:
            pattern_clarity = 1.0 - (similar_instances / total_instances)
            confidence_factors.append(pattern_clarity)
        
        # Factor 3: Distribution across ranges
        non_empty_ranges = len([pattern for pattern, instances in range_patterns.items() if instances])
        distribution_confidence = min(1.0, non_empty_ranges / 3.0)
        confidence_factors.append(distribution_confidence)
        
        return np.mean(confidence_factors)
    
    def _determine_pattern_type(
        self,
        instance_patterns: Dict[str, List[int]],
        range_patterns: Dict[str, List[int]],
        difficulty_patterns: Dict[str, List[int]],
        error_patterns: Dict[str, List[int]]
    ) -> str:
        """Determine the dominant type of complementarity pattern."""
        
        # Count significance of each pattern type
        pattern_scores = {}
        
        # Instance-specific patterns
        model1_better = len(instance_patterns.get('model1_better', []))
        model2_better = len(instance_patterns.get('model2_better', []))
        pattern_scores['instance_specific'] = model1_better + model2_better
        
        # Error patterns
        error_comp = len(error_patterns.get('model1_errors', [])) + len(error_patterns.get('model2_errors', []))
        pattern_scores['error_based'] = error_comp
        
        # Range patterns
        range_diversity = len([p for p, instances in range_patterns.items() if instances])
        pattern_scores['range_based'] = range_diversity
        
        # Difficulty patterns
        difficulty_diversity = len([p for p, instances in difficulty_patterns.items() if instances])
        pattern_scores['difficulty_based'] = difficulty_diversity
        
        # Return dominant pattern type
        if not pattern_scores:
            return 'unknown'
        
        return max(pattern_scores.keys(), key=lambda x: pattern_scores[x])
    
    def _default_pattern_analysis(self, reason: str) -> PatternAnalysis:
        """Return default pattern analysis when analysis fails."""
        
        return PatternAnalysis(
            strength_score=0.0,
            instance_patterns={},
            complementary_regions=[],
            pattern_confidence=0.0,
            pattern_type=f"default ({reason})"
        )


class EnsemblePerformancePredictor:
    """Predict ensemble performance from individual model scores."""
    
    def __init__(self):
        self.prediction_cache: Dict[str, EnsembleAnalysis] = {}
        
    def predict_ensemble_performance(
        self,
        scores1: Dict[int, float],
        scores2: Dict[int, float],
        instance_metadata: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> EnsembleAnalysis:
        """Predict ensemble performance using multiple strategies."""
        
        try:
            # Get common instances
            common_instances = list(set(scores1.keys()) & set(scores2.keys()))
            if len(common_instances) < 2:
                return self._default_ensemble_analysis("Insufficient data for ensemble prediction")
            
            # Strategy 1: Simple averaging
            avg_prediction = self._predict_average_ensemble(scores1, scores2, common_instances)
            
            # Strategy 2: Weighted ensemble optimization
            weighted_prediction = self._predict_weighted_ensemble(scores1, scores2, common_instances)
            
            # Strategy 3: Max ensemble (take best of both)
            max_prediction = self._predict_max_ensemble(scores1, scores2, common_instances)
            
            # Strategy 4: Conditional ensemble (context-dependent)
            conditional_prediction = self._predict_conditional_ensemble(
                scores1, scores2, common_instances, instance_metadata
            )
            
            # Select best prediction strategy
            best_prediction = self._select_best_prediction(
                avg_prediction, weighted_prediction, max_prediction, conditional_prediction
            )
            
            # Calculate synergy score
            synergy_score = self._calculate_synergy_score(
                scores1, scores2, common_instances, best_prediction['expected_improvement']
            )
            
            # Calculate confidence interval
            confidence_interval = self._calculate_ensemble_confidence_interval(
                scores1, scores2, common_instances, best_prediction
            )
            
            return EnsembleAnalysis(
                expected_improvement=best_prediction['expected_improvement'],
                confidence_interval=confidence_interval,
                optimal_weights=best_prediction['optimal_weights'],
                synergy_score=synergy_score,
                prediction_method=best_prediction['method']
            )
            
        except Exception as e:
            logger.debug(f"Ensemble prediction failed: {e}")
            return self._default_ensemble_analysis(f"Ensemble prediction error: {str(e)}")
    
    def _predict_average_ensemble(
        self,
        scores1: Dict[int, float],
        scores2: Dict[int, float],
        common_instances: List[int]
    ) -> Dict[str, Any]:
        """Predict performance using simple averaging."""
        
        individual_scores1 = [scores1[i] for i in common_instances]
        individual_scores2 = [scores2[i] for i in common_instances]
        ensemble_scores = [(s1 + s2) / 2 for s1, s2 in zip(individual_scores1, individual_scores2)]
        
        # Calculate improvement over best individual model
        best_individual = [max(s1, s2) for s1, s2 in zip(individual_scores1, individual_scores2)]
        improvements = [ens - best for ens, best in zip(ensemble_scores, best_individual)]
        
        expected_improvement = np.mean(improvements)
        
        return {
            'expected_improvement': expected_improvement,
            'optimal_weights': {'model1': 0.5, 'model2': 0.5},
            'method': 'simple_average',
            'ensemble_scores': ensemble_scores
        }
    
    def _predict_weighted_ensemble(
        self,
        scores1: Dict[int, float],
        scores2: Dict[int, float],
        common_instances: List[int]
    ) -> Dict[str, Any]:
        """Predict performance using optimized weighted ensemble."""
        
        individual_scores1 = [scores1[i] for i in common_instances]
        individual_scores2 = [scores2[i] for i in common_instances]
        
        # Find optimal weights using grid search
        best_improvement = -float('inf')
        best_weights = {'model1': 0.5, 'model2': 0.5}
        best_ensemble_scores = []
        
        # Test different weight combinations
        for w1 in np.arange(0.0, 1.1, 0.1):
            w2 = 1.0 - w1
            
            ensemble_scores = [w1 * s1 + w2 * s2 for s1, s2 in zip(individual_scores1, individual_scores2)]
            best_individual = [max(s1, s2) for s1, s2 in zip(individual_scores1, individual_scores2)]
            improvements = [ens - best for ens, best in zip(ensemble_scores, best_individual)]
            
            avg_improvement = np.mean(improvements)
            
            if avg_improvement > best_improvement:
                best_improvement = avg_improvement
                best_weights = {'model1': w1, 'model2': w2}  
                best_ensemble_scores = ensemble_scores
        
        return {
            'expected_improvement': best_improvement,
            'optimal_weights': best_weights,
            'method': 'weighted_ensemble',
            'ensemble_scores': best_ensemble_scores
        }
    
    def _predict_max_ensemble(
        self,
        scores1: Dict[int, float],
        scores2: Dict[int, float],
        common_instances: List[int]
    ) -> Dict[str, Any]:
        """Predict performance using max ensemble (oracle selection)."""
        
        individual_scores1 = [scores1[i] for i in common_instances]
        individual_scores2 = [scores2[i] for i in common_instances]
        
        # Take maximum of both models for each instance
        ensemble_scores = [max(s1, s2) for s1, s2 in zip(individual_scores1, individual_scores2)]
        
        # Calculate improvement over average individual performance
        avg_individual = [(s1 + s2) / 2 for s1, s2 in zip(individual_scores1, individual_scores2)]
        improvements = [ens - avg for ens, avg in zip(ensemble_scores, avg_individual)]
        
        expected_improvement = np.mean(improvements)
        
        # Weights represent selection probability (simplified)
        model1_selected = sum(1 for s1, s2 in zip(individual_scores1, individual_scores2) if s1 >= s2)
        total_instances = len(common_instances)
        
        weights = {
            'model1': model1_selected / total_instances,
            'model2': 1.0 - (model1_selected / total_instances)
        }
        
        return {
            'expected_improvement': expected_improvement,
            'optimal_weights': weights,
            'method': 'max_ensemble',
            'ensemble_scores': ensemble_scores
        }
    
    def _predict_conditional_ensemble(
        self,
        scores1: Dict[int, float],
        scores2: Dict[int, float],
        common_instances: List[int],
        instance_metadata: Optional[Dict[int, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Predict performance using context-dependent ensemble."""
        
        if not instance_metadata:
            # Fallback to weighted ensemble if no metadata
            return self._predict_weighted_ensemble(scores1, scores2, common_instances)
        
        # Group instances by metadata characteristics
        grouped_instances = defaultdict(list)
        
        for instance_id in common_instances:
            if instance_id in instance_metadata:
                # Use difficulty as grouping criterion (could extend to other characteristics)
                difficulty = instance_metadata[instance_id].get('difficulty', 'medium')
                grouped_instances[difficulty].append(instance_id)
            else:
                grouped_instances['unknown'].append(instance_id)
        
        # Find optimal weights for each group
        group_weights = {}
        ensemble_scores = []
        
        for group, group_instances in grouped_instances.items():
            if len(group_instances) >= 2:
                # Find optimal weights for this group
                group_scores1 = [scores1[i] for i in group_instances]
                group_scores2 = [scores2[i] for i in group_instances]
                
                group_prediction = self._predict_weighted_ensemble(
                    {i: scores1[i] for i in group_instances},
                    {i: scores2[i] for i in group_instances},
                    group_instances
                )
                
                group_weights[group] = group_prediction['optimal_weights']
                ensemble_scores.extend(group_prediction['ensemble_scores'])
            else:
                # Use default weights for small groups
                group_weights[group] = {'model1': 0.5, 'model2': 0.5}
                for instance_id in group_instances:
                    ens_score = (scores1[instance_id] + scores2[instance_id]) / 2
                    ensemble_scores.append(ens_score)
        
        # Calculate overall improvement
        individual_scores1 = [scores1[i] for i in common_instances]
        individual_scores2 = [scores2[i] for i in common_instances]
        best_individual = [max(s1, s2) for s1, s2 in zip(individual_scores1, individual_scores2)]
        improvements = [ens - best for ens, best in zip(ensemble_scores, best_individual)]
        
        expected_improvement = np.mean(improvements)
        
        # Use average weights across groups
        avg_weight1 = np.mean([weights['model1'] for weights in group_weights.values()])
        avg_weight2 = 1.0 - avg_weight1
        
        return {
            'expected_improvement': expected_improvement,
            'optimal_weights': {'model1': avg_weight1, 'model2': avg_weight2},
            'method': 'conditional_ensemble',
            'ensemble_scores': ensemble_scores,
            'group_weights': group_weights
        }
    
    def _select_best_prediction(
        self,
        avg_pred: Dict[str, Any],
        weighted_pred: Dict[str, Any],
        max_pred: Dict[str, Any],
        conditional_pred: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Select the best prediction strategy."""
        
        predictions = [avg_pred, weighted_pred, max_pred, conditional_pred]
        
        # Select based on expected improvement
        best_prediction = max(predictions, key=lambda x: x['expected_improvement'])
        
        return best_prediction
    
    def _calculate_synergy_score(
        self,
        scores1: Dict[int, float],
        scores2: Dict[int, float],
        common_instances: List[int],
        expected_improvement: float
    ) -> float:
        """Calculate synergy score between models."""
        
        # Synergy is the improvement beyond what would be expected from random combination
        individual_scores1 = [scores1[i] for i in common_instances]
        individual_scores2 = [scores2[i] for i in common_instances]
        
        # Expected improvement from random combination (baseline)
        random_baseline = 0.0  # No expected improvement from random combination
        
        # Synergy is the improvement above baseline
        synergy = expected_improvement - random_baseline
        
        # Normalize to [0, 1] range
        max_possible_synergy = 1.0 - max(np.mean(individual_scores1), np.mean(individual_scores2))
        
        if max_possible_synergy > 0:
            normalized_synergy = synergy / max_possible_synergy
        else:
            normalized_synergy = 0.0
        
        return max(0.0, min(1.0, normalized_synergy))
    
    def _calculate_ensemble_confidence_interval(
        self,
        scores1: Dict[int, float],
        scores2: Dict[int, float],
        common_instances: List[int],
        prediction: Dict[str, Any],
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for ensemble improvement."""
        
        try:
            ensemble_scores = prediction['ensemble_scores']
            individual_scores1 = [scores1[i] for i in common_instances]
            individual_scores2 = [scores2[i] for i in common_instances]
            
            # Calculate improvements
            best_individual = [max(s1, s2) for s1, s2 in zip(individual_scores1, individual_scores2)]
            improvements = [ens - best for ens, best in zip(ensemble_scores, best_individual)]
            
            if len(improvements) < 2:
                return (0.0, 0.0)
            
            # Calculate confidence interval for mean improvement
            mean_improvement = np.mean(improvements)
            std_improvement = np.std(improvements, ddof=1)
            n = len(improvements)
            
            # t-distribution critical value
            alpha = 1 - confidence_level
            df = n - 1
            t_critical = stats.t.ppf(1 - alpha/2, df)
            
            margin_of_error = t_critical * (std_improvement / np.sqrt(n))
            
            lower_bound = mean_improvement - margin_of_error
            upper_bound = mean_improvement + margin_of_error
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            logger.debug(f"Confidence interval calculation failed: {e}")
            return (0.0, 0.0)
    
    def _default_ensemble_analysis(self, reason: str) -> EnsembleAnalysis:
        """Return default ensemble analysis when prediction fails."""
        
        return EnsembleAnalysis(
            expected_improvement=0.0,
            confidence_interval=(0.0, 0.0),
            optimal_weights={'model1': 0.5, 'model2': 0.5},
            synergy_score=0.0,
            prediction_method=f"default ({reason})"
        )


class AdvancedComplementarityAnalyzer:
    """Statistical analysis of performance complementarity replacing simple heuristics."""
    
    def __init__(self):
        self.statistical_tester = StatisticalSignificanceTester()
        self.pattern_analyzer = PerformancePatternAnalyzer()
        self.ensemble_predictor = EnsemblePerformancePredictor()
        self.analysis_history: List[ComplementarityAnalysis] = []
        
    def analyze_complementarity(
        self,
        parent1_scores: Dict[int, float],
        parent2_scores: Dict[int, float],
        instance_metadata: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> ComplementarityAnalysis:
        """Deep analysis of performance complementarity."""
        
        try:
            # Analysis 1: Statistical Complementarity
            stat_analysis = self.statistical_tester.test_complementarity(
                parent1_scores, parent2_scores
            )
            
            # Analysis 2: Pattern-based Complementarity  
            pattern_analysis = self.pattern_analyzer.find_complementary_patterns(
                parent1_scores, parent2_scores, instance_metadata
            )
            
            # Analysis 3: Ensemble Performance Prediction
            ensemble_analysis = self.ensemble_predictor.predict_ensemble_performance(
                parent1_scores, parent2_scores, instance_metadata
            )
            
            # Combine analyses into overall complementarity score
            overall_score = self._calculate_overall_complementarity(
                stat_analysis, pattern_analysis, ensemble_analysis
            )
            
            # Calculate analysis confidence
            analysis_confidence = self._calculate_analysis_confidence(
                parent1_scores, parent2_scores, stat_analysis, pattern_analysis, ensemble_analysis
            )
            
            # Generate explanation
            explanation = self._generate_complementarity_explanation(
                stat_analysis, pattern_analysis, ensemble_analysis, overall_score
            )
            
            result = ComplementarityAnalysis(
                statistical_significance=stat_analysis.significance_level,
                complementarity_strength=pattern_analysis.strength_score,
                predicted_ensemble_gain=ensemble_analysis.expected_improvement,
                confidence_interval=ensemble_analysis.confidence_interval,
                complementary_instance_patterns=pattern_analysis.instance_patterns,
                overall_score=overall_score,
                analysis_confidence=analysis_confidence,
                explanation=explanation
            )
            
            # Record analysis for learning
            self.analysis_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Complementarity analysis failed: {e}")
            return self._default_complementarity_analysis(f"Analysis error: {str(e)}")
    
    def _calculate_overall_complementarity(
        self,
        stat_analysis: StatisticalAnalysis,
        pattern_analysis: PatternAnalysis,
        ensemble_analysis: EnsembleAnalysis
    ) -> float:
        """Calculate overall complementarity score."""
        
        # Weight different aspects of complementarity
        weights = {
            'statistical': 0.3,
            'pattern': 0.4,
            'ensemble': 0.3
        }
        
        # Normalize scores to [0, 1] range
        stat_score = min(1.0, stat_analysis.significance_level)
        pattern_score = pattern_analysis.strength_score
        ensemble_score = max(0.0, min(1.0, ensemble_analysis.expected_improvement * 2))  # Scale improvement
        
        overall_score = (
            weights['statistical'] * stat_score +
            weights['pattern'] * pattern_score +
            weights['ensemble'] * ensemble_score
        )
        
        return max(0.0, min(1.0, overall_score))
    
    def _calculate_analysis_confidence(
        self,
        parent1_scores: Dict[int, float],
        parent2_scores: Dict[int, float],
        stat_analysis: StatisticalAnalysis,
        pattern_analysis: PatternAnalysis,
        ensemble_analysis: EnsembleAnalysis
    ) -> float:
        """Calculate confidence in the complementarity analysis."""
        
        confidence_factors = []
        
        # Factor 1: Sample size
        common_instances = len(set(parent1_scores.keys()) & set(parent2_scores.keys()))
        if common_instances >= 20:
            size_confidence = 1.0
        elif common_instances >= 10:
            size_confidence = 0.8
        elif common_instances >= 5:
            size_confidence = 0.6
        else:
            size_confidence = 0.3
        
        confidence_factors.append(size_confidence)
        
        # Factor 2: Statistical test reliability
        if stat_analysis.test_method.startswith('default'):
            stat_confidence = 0.2
        elif stat_analysis.p_value < 0.05:
            stat_confidence = 0.9
        elif stat_analysis.p_value < 0.1:
            stat_confidence = 0.7
        else:
            stat_confidence = 0.5
        
        confidence_factors.append(stat_confidence)
        
        # Factor 3: Pattern analysis confidence
        confidence_factors.append(pattern_analysis.pattern_confidence)
        
        # Factor 4: Ensemble prediction reliability
        ci_width = ensemble_analysis.confidence_interval[1] - ensemble_analysis.confidence_interval[0]
        if ci_width < 0.1:
            ensemble_confidence = 0.9
        elif ci_width < 0.2:
            ensemble_confidence = 0.7
        else:
            ensemble_confidence = 0.5
        
        confidence_factors.append(ensemble_confidence)
        
        return np.mean(confidence_factors)
    
    def _generate_complementarity_explanation(
        self,
        stat_analysis: StatisticalAnalysis,
        pattern_analysis: PatternAnalysis,
        ensemble_analysis: EnsembleAnalysis,
        overall_score: float
    ) -> str:
        """Generate human-readable explanation of complementarity analysis."""
        
        explanations = []
        
        # Statistical significance explanation
        if stat_analysis.significance_level > 0.8:
            explanations.append(f"Statistically significant complementarity (p={stat_analysis.p_value:.3f})")
        elif stat_analysis.significance_level > 0.5:
            explanations.append("Moderate statistical evidence of complementarity")
        else:
            explanations.append("Limited statistical evidence of complementarity")
        
        # Pattern explanation
        if pattern_analysis.strength_score > 0.7:
            explanations.append(f"Strong {pattern_analysis.pattern_type} complementarity patterns")
        elif pattern_analysis.strength_score > 0.4:
            explanations.append(f"Moderate {pattern_analysis.pattern_type} patterns detected")
        else:
            explanations.append("Weak complementarity patterns")
        
        # Ensemble performance explanation
        if ensemble_analysis.expected_improvement > 0.1:
            explanations.append(f"Expected ensemble improvement: {ensemble_analysis.expected_improvement:.3f}")
        elif ensemble_analysis.expected_improvement > 0.05:
            explanations.append("Modest ensemble performance gain expected")
        else:
            explanations.append("Limited ensemble benefit predicted")
        
        # Overall assessment
        if overall_score > 0.7:
            explanations.append("STRONG overall complementarity")
        elif overall_score > 0.5:
            explanations.append("MODERATE overall complementarity")
        else:
            explanations.append("WEAK overall complementarity")
        
        return "; ".join(explanations)
    
    def _default_complementarity_analysis(self, reason: str) -> ComplementarityAnalysis:
        """Return default complementarity analysis when analysis fails."""
        
        return ComplementarityAnalysis(
            statistical_significance=0.5,
            complementarity_strength=0.3,
            predicted_ensemble_gain=0.0,
            confidence_interval=(0.0, 0.0),
            complementary_instance_patterns={},
            overall_score=0.3,
            analysis_confidence=0.2,
            explanation=f"Default analysis ({reason})"
        )
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get statistics about complementarity analysis performance."""
        
        if not self.analysis_history:
            return {"message": "No analysis history available"}
        
        recent_analyses = self.analysis_history[-20:]
        
        return {
            "total_analyses": len(self.analysis_history),
            "recent_avg_scores": {
                "statistical_significance": np.mean([a.statistical_significance for a in recent_analyses]),
                "complementarity_strength": np.mean([a.complementarity_strength for a in recent_analyses]),
                "predicted_ensemble_gain": np.mean([a.predicted_ensemble_gain for a in recent_analyses]),
                "overall_score": np.mean([a.overall_score for a in recent_analyses])
            },
            "average_confidence": np.mean([a.analysis_confidence for a in recent_analyses]),
            "high_complementarity_rate": np.mean([a.overall_score > 0.7 for a in recent_analyses])
        } 