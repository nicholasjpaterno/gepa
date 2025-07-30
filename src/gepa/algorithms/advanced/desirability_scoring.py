"""Advanced desirability scoring with multi-criteria decision analysis and risk assessment."""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class RiskFactor(Enum):
    """Different types of risks in merge operations."""
    COMPLEXITY = "complexity"
    COMPATIBILITY = "compatibility"
    PERFORMANCE = "performance"
    MAINTENANCE = "maintenance"
    STABILITY = "stability"


class DecisionCriterion(Enum):
    """Criteria for desirability scoring."""
    COMPATIBILITY = "compatibility"
    COMPLEMENTARITY = "complementarity"
    PERFORMANCE_GAIN = "performance_gain"
    RISK_FACTOR = "risk_factor"
    DIVERSITY_BENEFIT = "diversity_benefit"
    COST_EFFICIENCY = "cost_efficiency"


@dataclass
class RiskAssessment:
    """Risk assessment result."""
    complexity_risk: float
    compatibility_risk: float
    performance_risk: float
    maintenance_risk: float
    stability_risk: float
    overall_risk: float
    high_risk_factors: List[str]
    risk_explanation: str


@dataclass
class MergeContext:
    """Context for merge desirability analysis."""
    parent1_id: str
    parent2_id: str
    system_complexity: int = 1
    merge_history: Dict[str, List[Dict[str, Any]]] = None
    optimization_phase: str = "exploration"  # exploration, exploitation, refinement
    budget_remaining: float = 1.0
    performance_targets: Dict[str, float] = None
    constraints: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.merge_history is None:
            self.merge_history = {}
        if self.performance_targets is None:
            self.performance_targets = {}
        if self.constraints is None:
            self.constraints = {}


@dataclass  
class DesirabilityScore:
    """Comprehensive desirability analysis result."""
    score: float
    confidence: float
    threshold: float
    is_desirable: bool
    criterion_scores: Dict[str, float]
    criterion_weights: Dict[str, float]
    risk_assessment: RiskAssessment
    explanation: str
    recommendation: str


class RiskAnalyzer:
    """Analyze risks associated with merge operations."""
    
    def __init__(self):
        self.risk_history: List[Dict[str, Any]] = []
        self.risk_models: Dict[RiskFactor, Any] = {}
        
    def assess_merge_risk(self, context: MergeContext) -> RiskAssessment:
        """Comprehensive risk assessment for merge operation."""
        
        try:
            # Risk Factor 1: Complexity Risk
            complexity_risk = self._assess_complexity_risk(context)
            
            # Risk Factor 2: Compatibility Risk
            compatibility_risk = self._assess_compatibility_risk(context)
            
            # Risk Factor 3: Performance Degradation Risk
            performance_risk = self._assess_performance_degradation_risk(context)
            
            # Risk Factor 4: Maintenance Risk
            maintenance_risk = self._assess_maintenance_risk(context)
            
            # Risk Factor 5: System Stability Risk
            stability_risk = self._assess_stability_risk(context)
            
            # Calculate overall risk (weighted combination)
            risk_weights = {
                RiskFactor.COMPLEXITY: 0.2,
                RiskFactor.COMPATIBILITY: 0.25,
                RiskFactor.PERFORMANCE: 0.25,
                RiskFactor.MAINTENANCE: 0.15,
                RiskFactor.STABILITY: 0.15
            }
            
            overall_risk = (
                risk_weights[RiskFactor.COMPLEXITY] * complexity_risk +
                risk_weights[RiskFactor.COMPATIBILITY] * compatibility_risk +
                risk_weights[RiskFactor.PERFORMANCE] * performance_risk +
                risk_weights[RiskFactor.MAINTENANCE] * maintenance_risk +
                risk_weights[RiskFactor.STABILITY] * stability_risk
            )
            
            # Identify high-risk factors
            risk_values = {
                'complexity': complexity_risk,
                'compatibility': compatibility_risk,
                'performance': performance_risk,
                'maintenance': maintenance_risk,
                'stability': stability_risk
            }
            
            high_risk_factors = [
                factor for factor, risk in risk_values.items() if risk > 0.7
            ]
            
            # Generate risk explanation
            risk_explanation = self._generate_risk_explanation(
                risk_values, high_risk_factors, overall_risk
            )
            
            assessment = RiskAssessment(
                complexity_risk=complexity_risk,
                compatibility_risk=compatibility_risk,
                performance_risk=performance_risk,
                maintenance_risk=maintenance_risk,
                stability_risk=stability_risk,
                overall_risk=overall_risk,
                high_risk_factors=high_risk_factors,
                risk_explanation=risk_explanation
            )
            
            # Record assessment for learning
            self.risk_history.append({
                'context': context,
                'assessment': assessment,
                'timestamp': 'now'  # Would use actual timestamp
            })
            
            return assessment
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return self._default_risk_assessment(f"Risk assessment error: {str(e)}")
    
    def _assess_complexity_risk(self, context: MergeContext) -> float:
        """Assess risk related to system complexity increase."""
        
        # Factor 1: Base system complexity
        base_complexity = min(1.0, context.system_complexity / 10.0)  # Normalize to [0,1]
        
        # Factor 2: Merge complexity (combining different parents)
        if context.parent1_id == context.parent2_id:
            merge_complexity = 0.0  # No complexity from identical parents
        else:
            merge_complexity = 0.3  # Base complexity from mixing different systems
        
        # Factor 3: Historical merge success rate
        historical_complexity = self._get_historical_complexity_risk(context)
        
        # Factor 4: Number of modules being merged
        # Simplified - would analyze actual module count
        module_complexity = 0.2  # Default moderate complexity
        
        # Combine factors
        complexity_risk = (
            0.3 * base_complexity +
            0.3 * merge_complexity +
            0.2 * historical_complexity +
            0.2 * module_complexity
        )
        
        return max(0.0, min(1.0, complexity_risk))
    
    def _assess_compatibility_risk(self, context: MergeContext) -> float:
        """Assess risk related to compatibility issues."""
        
        # Factor 1: Parent system similarity
        if context.parent1_id == context.parent2_id:
            similarity_risk = 0.0  # Identical systems are fully compatible
        else:
            # Would use actual compatibility analysis here
            similarity_risk = 0.4  # Default moderate risk for different systems
        
        # Factor 2: Historical compatibility issues
        historical_risk = self._get_historical_compatibility_risk(context)
        
        # Factor 3: System architecture complexity
        architecture_risk = min(1.0, context.system_complexity / 15.0)
        
        # Factor 4: Integration complexity
        integration_risk = 0.3  # Default moderate integration risk
        
        # Combine factors
        compatibility_risk = (
            0.4 * similarity_risk +
            0.3 * historical_risk +
            0.2 * architecture_risk +
            0.1 * integration_risk
        )
        
        return max(0.0, min(1.0, compatibility_risk))
    
    def _assess_performance_degradation_risk(self, context: MergeContext) -> float:
        """Assess risk of performance degradation after merge."""
        
        # Factor 1: Performance target constraints
        performance_risk = 0.0
        if context.performance_targets:
            # Higher risk if we have strict performance targets
            target_strictness = len(context.performance_targets) / 5.0  # Assume max 5 targets
            performance_risk += min(1.0, target_strictness * 0.3)
        
        # Factor 2: Historical performance after merges
        historical_perf_risk = self._get_historical_performance_risk(context)
        
        # Factor 3: Resource overhead from merging
        resource_risk = 0.2  # Default moderate overhead
        
        # Factor 4: Optimization phase consideration
        if context.optimization_phase == "exploitation":
            # Higher risk during exploitation phase (less tolerance for degradation)
            phase_risk = 0.4
        elif context.optimization_phase == "refinement":
            phase_risk = 0.6  # Highest risk during refinement
        else:
            phase_risk = 0.2  # Lower risk during exploration
        
        # Combine factors
        perf_degradation_risk = (
            0.3 * performance_risk +
            0.3 * historical_perf_risk +
            0.2 * resource_risk +
            0.2 * phase_risk
        )
        
        return max(0.0, min(1.0, perf_degradation_risk))
    
    def _assess_maintenance_risk(self, context: MergeContext) -> float:
        """Assess risk related to increased maintenance burden."""
        
        # Factor 1: System complexity impact on maintenance
        complexity_maintenance_risk = min(1.0, context.system_complexity / 12.0)
        
        # Factor 2: Code/prompt coherence after merge
        coherence_risk = 0.3  # Default risk from reduced coherence
        
        # Factor 3: Debug difficulty increase
        debug_risk = 0.25  # Default risk from debugging merged systems
        
        # Factor 4: Documentation and understanding difficulty
        documentation_risk = 0.2  # Default risk from increased documentation needs
        
        # Combine factors
        maintenance_risk = (
            0.4 * complexity_maintenance_risk +
            0.25 * coherence_risk +
            0.25 * debug_risk +
            0.1 * documentation_risk
        )
        
        return max(0.0, min(1.0, maintenance_risk))
    
    def _assess_stability_risk(self, context: MergeContext) -> float:
        """Assess risk to system stability from merge."""
        
        # Factor 1: System maturity (more mature = more stable)
        # Simplified - would analyze actual system age and testing
        maturity_risk = 0.3  # Default moderate risk
        
        # Factor 2: Testing coverage impact
        testing_risk = 0.4  # Default risk from potentially reduced test coverage
        
        # Factor 3: Edge case handling
        edge_case_risk = 0.35  # Default risk from edge case interactions
        
        # Factor 4: Rollback difficulty
        rollback_risk = 0.2  # Default risk if merge needs to be undone
        
        # Combine factors
        stability_risk = (
            0.3 * maturity_risk +
            0.3 * testing_risk +
            0.25 * edge_case_risk +
            0.15 * rollback_risk
        )
        
        return max(0.0, min(1.0, stability_risk))
    
    def _get_historical_complexity_risk(self, context: MergeContext) -> float:
        """Get complexity risk based on historical merge data."""
        
        if not self.risk_history:
            return 0.3  # Default moderate risk
        
        # Analyze similar historical merges
        similar_merges = [
            entry for entry in self.risk_history[-20:]  # Last 20 merges
            if abs(entry['context'].system_complexity - context.system_complexity) <= 2
        ]
        
        if not similar_merges:
            return 0.3
        
        # Average complexity risk from similar merges
        complexity_risks = [entry['assessment'].complexity_risk for entry in similar_merges]
        return np.mean(complexity_risks)
    
    def _get_historical_compatibility_risk(self, context: MergeContext) -> float:
        """Get compatibility risk based on historical data."""
        
        if not self.risk_history:
            return 0.4  # Default moderate risk
        
        # Look for merges involving same parent systems
        related_merges = [
            entry for entry in self.risk_history[-15:]
            if (entry['context'].parent1_id == context.parent1_id or
                entry['context'].parent2_id == context.parent2_id or
                entry['context'].parent1_id == context.parent2_id or
                entry['context'].parent2_id == context.parent1_id)
        ]
        
        if not related_merges:
            return 0.4
        
        compatibility_risks = [entry['assessment'].compatibility_risk for entry in related_merges]
        return np.mean(compatibility_risks)
    
    def _get_historical_performance_risk(self, context: MergeContext) -> float:
        """Get performance degradation risk based on historical data."""
        
        if not self.risk_history:
            return 0.35  # Default moderate risk
        
        # Analyze performance outcomes from recent merges
        recent_merges = self.risk_history[-10:]
        
        if not recent_merges:
            return 0.35
        
        performance_risks = [entry['assessment'].performance_risk for entry in recent_merges]
        return np.mean(performance_risks)
    
    def _generate_risk_explanation(
        self,
        risk_values: Dict[str, float],
        high_risk_factors: List[str],
        overall_risk: float
    ) -> str:
        """Generate human-readable risk explanation."""
        
        explanations = []
        
        # Overall risk level
        if overall_risk > 0.7:
            explanations.append("HIGH overall risk")
        elif overall_risk > 0.5:
            explanations.append("MODERATE overall risk")
        else:
            explanations.append("LOW overall risk")
        
        # Specific high-risk factors
        if high_risk_factors:
            explanations.append(f"High risk in: {', '.join(high_risk_factors)}")
        
        # Dominant risk factors
        sorted_risks = sorted(risk_values.items(), key=lambda x: x[1], reverse=True)
        top_risk = sorted_risks[0]
        
        if top_risk[1] > 0.6:
            explanations.append(f"Primary concern: {top_risk[0]} risk ({top_risk[1]:.2f})")
        
        return "; ".join(explanations)
    
    def _default_risk_assessment(self, reason: str) -> RiskAssessment:
        """Return default risk assessment when analysis fails."""
        
        return RiskAssessment(
            complexity_risk=0.5,
            compatibility_risk=0.5,
            performance_risk=0.5,
            maintenance_risk=0.5,
            stability_risk=0.5,
            overall_risk=0.5,
            high_risk_factors=[],
            risk_explanation=f"Default risk assessment ({reason})"
        )


class MultiCriteriaDecisionAnalysis:
    """Multi-criteria decision analysis for desirability scoring."""
    
    def __init__(self):
        self.decision_history: List[Dict[str, Any]] = []
        
    def calculate_score(
        self,
        criteria_scores: Dict[str, float],
        criteria_weights: Dict[str, float]
    ) -> float:
        """Calculate MCDA score using weighted sum method."""
        
        try:
            # Normalize weights to sum to 1
            total_weight = sum(criteria_weights.values())
            if total_weight == 0:
                return 0.0
            
            normalized_weights = {
                criterion: weight / total_weight
                for criterion, weight in criteria_weights.items()
            }
            
            # Calculate weighted score
            weighted_score = 0.0
            for criterion, score in criteria_scores.items():
                if criterion in normalized_weights:
                    weighted_score += normalized_weights[criterion] * score
            
            # Record decision for learning
            self.decision_history.append({
                'criteria_scores': criteria_scores.copy(),
                'criteria_weights': criteria_weights.copy(),
                'final_score': weighted_score
            })
            
            return max(0.0, min(1.0, weighted_score))
            
        except Exception as e:
            logger.debug(f"MCDA calculation failed: {e}")
            return 0.5
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get statistics about decision history."""
        
        if not self.decision_history:
            return {"message": "No decision history available"}
        
        recent_decisions = self.decision_history[-20:]
        
        return {
            "total_decisions": len(self.decision_history),
            "recent_avg_score": np.mean([d['final_score'] for d in recent_decisions]),
            "score_distribution": {
                "high_scores": len([d for d in recent_decisions if d['final_score'] > 0.7]),
                "medium_scores": len([d for d in recent_decisions if 0.3 <= d['final_score'] <= 0.7]), 
                "low_scores": len([d for d in recent_decisions if d['final_score'] < 0.3])
            }
        }


class AdvancedDesirabilityScorer:
    """Multi-criteria desirability analysis with learned preferences."""
    
    def __init__(self):
        self.risk_analyzer = RiskAnalyzer()
        self.mcda_solver = MultiCriteriaDecisionAnalysis()
        self.scoring_history: List[DesirabilityScore] = []
        
        # Base criterion weights (can be adapted over time)
        self.base_weights = {
            DecisionCriterion.COMPATIBILITY: 0.25,
            DecisionCriterion.COMPLEMENTARITY: 0.25,
            DecisionCriterion.PERFORMANCE_GAIN: 0.20,
            DecisionCriterion.RISK_FACTOR: 0.15,
            DecisionCriterion.DIVERSITY_BENEFIT: 0.10,
            DecisionCriterion.COST_EFFICIENCY: 0.05
        }
        
        # Learning parameters
        self.adaptation_rate = 0.1
        self.success_threshold = 0.7
        
    def calculate_desirability(
        self,
        compatibility_analysis: Any,  # CompatibilityAnalysis type
        complementarity_analysis: Any,  # ComplementarityAnalysis type  
        performance_impact: Dict[str, float],
        context: MergeContext
    ) -> DesirabilityScore:
        """Comprehensive desirability analysis using multiple criteria."""
        
        try:
            # Criterion 1: Compatibility
            compatibility_score = getattr(compatibility_analysis, 'overall_score', 0.5)
            
            # Criterion 2: Complementarity
            complementarity_score = getattr(complementarity_analysis, 'overall_score', 0.5)
            
            # Criterion 3: Performance gain
            performance_gain_score = self._calculate_performance_gain_score(performance_impact)
            
            # Criterion 4: Risk assessment
            risk_assessment = self.risk_analyzer.assess_merge_risk(context)
            risk_score = 1.0 - risk_assessment.overall_risk  # Convert risk to positive score
            
            # Criterion 5: Diversity benefit
            diversity_score = self._calculate_diversity_benefit(context)
            
            # Criterion 6: Cost efficiency
            cost_efficiency_score = self._calculate_cost_efficiency(context)
            
            # Combine criterion scores
            criteria_scores = {
                'compatibility': compatibility_score,
                'complementarity': complementarity_score,
                'performance_gain': performance_gain_score,
                'risk_factor': risk_score,
                'diversity_benefit': diversity_score,
                'cost_efficiency': cost_efficiency_score
            }
            
            # Adapt weights based on historical success
            adapted_weights = self._adapt_criterion_weights(context)
            
            # Calculate MCDA score
            mcda_score = self.mcda_solver.calculate_score(criteria_scores, adapted_weights)
            
            # Calculate confidence in the score
            confidence = self._estimate_confidence(
                compatibility_analysis, complementarity_analysis, context, risk_assessment
            )
            
            # Calculate adaptive threshold
            adaptive_threshold = self._calculate_adaptive_threshold(context)
            
            # Make desirability decision
            is_desirable = mcda_score > adaptive_threshold and confidence > 0.6
            
            # Generate explanation and recommendation
            explanation = self._generate_explanation(
                criteria_scores, adapted_weights, mcda_score, risk_assessment
            )
            
            recommendation = self._generate_recommendation(
                mcda_score, adaptive_threshold, confidence, risk_assessment, context
            )
            
            result = DesirabilityScore(
                score=mcda_score,
                confidence=confidence,
                threshold=adaptive_threshold,
                is_desirable=is_desirable,
                criterion_scores=criteria_scores,
                criterion_weights=adapted_weights,
                risk_assessment=risk_assessment,
                explanation=explanation,
                recommendation=recommendation
            )
            
            # Record score for learning
            self.scoring_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Desirability scoring failed: {e}")
            return self._default_desirability_score(f"Scoring error: {str(e)}")
    
    def _calculate_performance_gain_score(self, performance_impact: Dict[str, float]) -> float:
        """Calculate performance gain score from impact metrics."""
        
        if not performance_impact:
            return 0.5  # Default neutral score
        
        # Extract relevant performance metrics
        expected_gain = performance_impact.get('expected_gain', 0.0)
        confidence_lower = performance_impact.get('confidence_lower', 0.0)
        confidence_upper = performance_impact.get('confidence_upper', 0.0)
        
        # Base score from expected gain
        gain_score = max(0.0, min(1.0, expected_gain * 2))  # Scale to [0,1]
        
        # Adjust for confidence interval width (narrow interval = higher confidence)
        ci_width = confidence_upper - confidence_lower
        confidence_adjustment = max(0.0, 1.0 - ci_width)
        
        # Combine gain and confidence
        performance_score = 0.7 * gain_score + 0.3 * confidence_adjustment
        
        return max(0.0, min(1.0, performance_score))
    
    def _calculate_diversity_benefit(self, context: MergeContext) -> float:
        """Calculate diversity benefit score."""
        
        # Factor 1: Parent diversity (different parents = higher diversity)
        if context.parent1_id == context.parent2_id:
            parent_diversity = 0.0  # No diversity from identical parents
        else:
            parent_diversity = 1.0  # Maximum diversity from different parents
        
        # Factor 2: System portfolio diversity
        # Would analyze current system portfolio diversity in full implementation
        portfolio_diversity = 0.5  # Default moderate portfolio diversity
        
        # Factor 3: Exploration vs exploitation phase
        if context.optimization_phase == "exploration":
            phase_diversity_bonus = 0.3  # Higher value for diversity during exploration
        elif context.optimization_phase == "exploitation":
            phase_diversity_bonus = 0.1  # Lower value during exploitation
        else:  # refinement
            phase_diversity_bonus = 0.0  # No diversity bonus during refinement
        
        # Combine factors
        diversity_score = (
            0.5 * parent_diversity +
            0.3 * portfolio_diversity +
            0.2 * phase_diversity_bonus
        )
        
        return max(0.0, min(1.0, diversity_score))
    
    def _calculate_cost_efficiency(self, context: MergeContext) -> float:
        """Calculate cost efficiency score."""
        
        # Factor 1: Budget efficiency
        if context.budget_remaining > 0.8:
            budget_efficiency = 0.3  # Low efficiency when plenty of budget remains
        elif context.budget_remaining > 0.5:
            budget_efficiency = 0.7  # Good efficiency with moderate budget
        elif context.budget_remaining > 0.2:
            budget_efficiency = 1.0  # High efficiency when budget is limited
        else:
            budget_efficiency = 0.8  # Slightly lower when budget is very low (risky)
        
        # Factor 2: Merge operation cost (simplified)
        merge_cost = 0.1  # Default low cost for merge operation
        cost_factor = 1.0 - merge_cost
        
        # Factor 3: Expected ROI
        expected_roi = 0.6  # Default moderate ROI expectation
        
        # Combine factors
        cost_efficiency = (
            0.4 * budget_efficiency +
            0.3 * cost_factor +
            0.3 * expected_roi
        )
        
        return max(0.0, min(1.0, cost_efficiency))
    
    def _adapt_criterion_weights(self, context: MergeContext) -> Dict[str, float]:
        """Adapt criterion weights based on historical success."""
        
        if len(self.scoring_history) < 5:
            return self.base_weights.copy()  # Not enough history for adaptation
        
        # Analyze recent successful vs unsuccessful decisions
        recent_scores = self.scoring_history[-20:]  # Last 20 decisions
        
        successful_scores = [s for s in recent_scores if s.score > self.success_threshold]
        unsuccessful_scores = [s for s in recent_scores if s.score <= self.success_threshold]
        
        if not successful_scores or not unsuccessful_scores:
            return self.base_weights.copy()  # Can't adapt without both success and failure examples
        
        # Calculate average criterion scores for successful vs unsuccessful decisions
        adapted_weights = self.base_weights.copy()
        
        for criterion in DecisionCriterion:
            criterion_key = criterion.value
            
            if criterion_key in adapted_weights:
                # Get average scores for this criterion
                successful_avg = np.mean([
                    s.criterion_scores.get(criterion_key, 0.5) for s in successful_scores
                ])
                unsuccessful_avg = np.mean([
                    s.criterion_scores.get(criterion_key, 0.5) for s in unsuccessful_scores
                ])
                
                # Adjust weight based on discriminative power
                discriminative_power = abs(successful_avg - unsuccessful_avg)
                
                # Increase weight if criterion discriminates well between success/failure
                weight_adjustment = self.adaptation_rate * discriminative_power
                
                if successful_avg > unsuccessful_avg:
                    # Criterion correlates with success - increase weight
                    adapted_weights[criterion_key] = min(0.8, 
                        adapted_weights[criterion_key] + weight_adjustment)
                else:
                    # Criterion correlates with failure - decrease weight
                    adapted_weights[criterion_key] = max(0.05, 
                        adapted_weights[criterion_key] - weight_adjustment)
        
        # Renormalize weights
        total_weight = sum(adapted_weights.values())
        adapted_weights = {k: v / total_weight for k, v in adapted_weights.items()}
        
        return adapted_weights
    
    def _estimate_confidence(
        self,
        compatibility_analysis: Any,
        complementarity_analysis: Any,
        context: MergeContext,
        risk_assessment: RiskAssessment
    ) -> float:
        """Estimate confidence in desirability score."""
        
        confidence_factors = []
        
        # Factor 1: Compatibility analysis confidence
        compatibility_confidence = getattr(compatibility_analysis, 'confidence', 0.5)
        confidence_factors.append(compatibility_confidence)
        
        # Factor 2: Complementarity analysis confidence
        complementarity_confidence = getattr(complementarity_analysis, 'analysis_confidence', 0.5)
        confidence_factors.append(complementarity_confidence)
        
        # Factor 3: Risk assessment reliability
        if len(risk_assessment.high_risk_factors) == 0:
            risk_confidence = 0.8  # High confidence when no high-risk factors
        elif len(risk_assessment.high_risk_factors) <= 2:
            risk_confidence = 0.6  # Moderate confidence with few high-risk factors
        else:
            risk_confidence = 0.4  # Lower confidence with many high-risk factors
        
        confidence_factors.append(risk_confidence)
        
        # Factor 4: Historical scoring confidence
        if len(self.scoring_history) >= 10:
            # Analyze consistency of recent decisions
            recent_scores = [s.score for s in self.scoring_history[-10:]]
            score_variance = np.var(recent_scores)
            consistency_confidence = max(0.3, 1.0 - score_variance)
        else:
            consistency_confidence = 0.5  # Default for insufficient history
        
        confidence_factors.append(consistency_confidence)
        
        # Factor 5: Context completeness
        context_completeness = self._assess_context_completeness(context)
        confidence_factors.append(context_completeness)
        
        return np.mean(confidence_factors)
    
    def _calculate_adaptive_threshold(self, context: MergeContext) -> float:
        """Calculate adaptive threshold based on context and history."""
        
        base_threshold = 0.6  # Base desirability threshold
        
        # Adjust based on optimization phase
        if context.optimization_phase == "exploration":
            phase_adjustment = -0.1  # Lower threshold during exploration
        elif context.optimization_phase == "exploitation":
            phase_adjustment = 0.0  # Normal threshold during exploitation
        else:  # refinement
            phase_adjustment = 0.1  # Higher threshold during refinement
        
        # Adjust based on budget remaining
        if context.budget_remaining < 0.3:
            budget_adjustment = 0.1  # Higher threshold when budget is low
        elif context.budget_remaining > 0.8:
            budget_adjustment = -0.05  # Slightly lower threshold with plenty of budget
        else:
            budget_adjustment = 0.0
        
        # Adjust based on historical success rate
        if len(self.scoring_history) >= 10:
            recent_success_rate = np.mean([
                s.is_desirable for s in self.scoring_history[-10:]
            ])
            
            if recent_success_rate > 0.8:
                success_adjustment = 0.05  # Raise threshold if too many successes
            elif recent_success_rate < 0.3:
                success_adjustment = -0.05  # Lower threshold if too few successes
            else:
                success_adjustment = 0.0
        else:
            success_adjustment = 0.0
        
        # Combine adjustments
        adaptive_threshold = base_threshold + phase_adjustment + budget_adjustment + success_adjustment
        
        return max(0.3, min(0.9, adaptive_threshold))  # Clamp to reasonable range
    
    def _assess_context_completeness(self, context: MergeContext) -> float:
        """Assess how complete the merge context information is."""
        
        completeness_factors = []
        
        # Factor 1: Parent ID availability
        if context.parent1_id and context.parent2_id:
            completeness_factors.append(1.0)
        else:
            completeness_factors.append(0.3)
        
        # Factor 2: System complexity information
        if context.system_complexity > 0:
            completeness_factors.append(0.8)
        else:
            completeness_factors.append(0.4)
        
        # Factor 3: Performance targets availability
        if context.performance_targets:
            completeness_factors.append(0.9)
        else:
            completeness_factors.append(0.5)
        
        # Factor 4: Merge history availability
        if context.merge_history:
            completeness_factors.append(0.8)
        else:
            completeness_factors.append(0.4)
        
        # Factor 5: Budget information
        if context.budget_remaining is not None:
            completeness_factors.append(0.7)
        else:
            completeness_factors.append(0.3)
        
        return np.mean(completeness_factors)
    
    def _generate_explanation(
        self,
        criteria_scores: Dict[str, float],
        criteria_weights: Dict[str, float],
        mcda_score: float,
        risk_assessment: RiskAssessment
    ) -> str:
        """Generate human-readable explanation of desirability analysis."""
        
        explanations = []
        
        # Overall score assessment
        if mcda_score > 0.8:
            explanations.append("HIGHLY desirable merge")
        elif mcda_score > 0.6:
            explanations.append("MODERATELY desirable merge")
        elif mcda_score > 0.4:
            explanations.append("MARGINALLY desirable merge")
        else:
            explanations.append("UNDESIRABLE merge")
        
        # Top contributing factors
        weighted_contributions = {
            criterion: score * criteria_weights.get(criterion, 0.0)
            for criterion, score in criteria_scores.items()
        }
        
        top_factors = sorted(weighted_contributions.items(), key=lambda x: x[1], reverse=True)[:2]
        
        factor_explanations = []
        for factor, contribution in top_factors:
            if contribution > 0.15:  # Significant contribution
                factor_explanations.append(f"{factor} ({contribution:.2f})")
        
        if factor_explanations:
            explanations.append(f"Key factors: {', '.join(factor_explanations)}")
        
        # Risk considerations
        if risk_assessment.overall_risk > 0.7:
            explanations.append(f"HIGH RISK: {risk_assessment.risk_explanation}")
        elif risk_assessment.overall_risk > 0.5:
            explanations.append(f"Moderate risk: {risk_assessment.risk_explanation}")
        
        return "; ".join(explanations)
    
    def _generate_recommendation(
        self,
        mcda_score: float,
        adaptive_threshold: float,
        confidence: float,
        risk_assessment: RiskAssessment,
        context: MergeContext
    ) -> str:
        """Generate actionable recommendation."""
        
        if mcda_score > adaptive_threshold and confidence > 0.7:
            if risk_assessment.overall_risk < 0.3:
                return "PROCEED with merge - high desirability, low risk"
            elif risk_assessment.overall_risk < 0.6:
                return "PROCEED CAUTIOUSLY - desirable but moderate risk"
            else:
                return "CONSIDER ALTERNATIVES - desirable but high risk"
        
        elif mcda_score > adaptive_threshold and confidence > 0.5:
            return "PROCEED WITH MONITORING - moderately confident in desirability"
        
        elif mcda_score > adaptive_threshold - 0.1:  # Close to threshold
            if context.optimization_phase == "exploration":
                return "PROCEED - acceptable for exploration phase"
            else:
                return "MARGINAL - consider if better alternatives unavailable"
        
        else:
            if risk_assessment.overall_risk > 0.7:
                return "REJECT - low desirability and high risk"
            else:
                return "REJECT - insufficient desirability score"
    
    def _default_desirability_score(self, reason: str) -> DesirabilityScore:
        """Return default desirability score when analysis fails."""
        
        default_risk = RiskAssessment(
            complexity_risk=0.5,
            compatibility_risk=0.5,
            performance_risk=0.5,
            maintenance_risk=0.5,
            stability_risk=0.5,
            overall_risk=0.5,
            high_risk_factors=[],
            risk_explanation=f"Default risk assessment ({reason})"
        )
        
        return DesirabilityScore(
            score=0.5,
            confidence=0.3,
            threshold=0.6,
            is_desirable=False,
            criterion_scores={'default': 0.5},
            criterion_weights={'default': 1.0},
            risk_assessment=default_risk,
            explanation=f"Default desirability analysis ({reason})",
            recommendation="INSUFFICIENT DATA - gather more information before proceeding"
        )
    
    def get_scoring_statistics(self) -> Dict[str, Any]:
        """Get statistics about desirability scoring performance."""
        
        if not self.scoring_history:
            return {"message": "No scoring history available"}
        
        recent_scores = self.scoring_history[-30:]
        
        return {
            "total_scores": len(self.scoring_history),
            "recent_desirability_rate": np.mean([s.is_desirable for s in recent_scores]),
            "recent_avg_score": np.mean([s.score for s in recent_scores]),
            "recent_avg_confidence": np.mean([s.confidence for s in recent_scores]),
            "recent_avg_threshold": np.mean([s.threshold for s in recent_scores]),
            "current_weights": self.base_weights,
            "high_risk_decisions": len([s for s in recent_scores if s.risk_assessment.overall_risk > 0.7])
        } 