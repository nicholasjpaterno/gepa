"""Intelligent module selection replacing round-robin heuristics."""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class SelectionCriterion(Enum):
    """Different criteria for module selection."""
    IMPROVEMENT_POTENTIAL = "improvement_potential"
    PERFORMANCE_BOTTLENECK = "performance_bottleneck" 
    MUTATION_SUCCESS_RATE = "mutation_success_rate"
    EXPLORATION_EXPLOITATION = "exploration_exploitation"


@dataclass
class ModuleAnalysis:
    """Analysis results for a module."""
    module_id: str
    improvement_potential: float
    bottleneck_score: float
    mutation_success_rate: float
    bandit_score: float
    overall_score: float
    selection_reasons: List[str]


@dataclass
class SelectionContext:
    """Context for intelligent module selection."""
    performance_history: Dict[str, List[float]]
    mutation_history: Dict[str, List[bool]]  # success/failure history
    optimization_phase: str = "exploration"  # exploration, exploitation, refinement
    budget_remaining: float = 1.0
    current_generation: int = 0


class MultiArmedBandit:
    """Multi-armed bandit for exploration-exploitation balance."""
    
    def __init__(self, exploration_factor: float = 1.4):
        self.exploration_factor = exploration_factor  # UCB1 exploration parameter
        self.arm_counts: Dict[str, int] = defaultdict(int)
        self.arm_rewards: Dict[str, List[float]] = defaultdict(list)
        self.total_selections = 0
        
    def select_arm(self, available_arms: List[str]) -> str:
        """Select arm using Upper Confidence Bound (UCB1) algorithm."""
        
        if self.total_selections == 0:
            # First selection - choose randomly
            return np.random.choice(available_arms)
        
        # Ensure all arms have been tried at least once
        untried_arms = [arm for arm in available_arms if self.arm_counts[arm] == 0]
        if untried_arms:
            return np.random.choice(untried_arms)
        
        # Calculate UCB1 scores for all arms
        ucb_scores = {}
        for arm in available_arms:
            if self.arm_counts[arm] == 0:
                ucb_scores[arm] = float('inf')
            else:
                avg_reward = np.mean(self.arm_rewards[arm])
                confidence_interval = self.exploration_factor * np.sqrt(
                    np.log(self.total_selections) / self.arm_counts[arm]
                )
                ucb_scores[arm] = avg_reward + confidence_interval
        
        # Select arm with highest UCB1 score
        return max(available_arms, key=lambda arm: ucb_scores[arm])
    
    def update_reward(self, arm: str, reward: float) -> None:
        """Update bandit with reward for selected arm."""
        self.arm_counts[arm] += 1
        self.arm_rewards[arm].append(reward)
        self.total_selections += 1
        
        # Keep reward history manageable
        if len(self.arm_rewards[arm]) > 100:
            self.arm_rewards[arm] = self.arm_rewards[arm][-100:]
    
    def get_arm_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all arms."""
        stats = {}
        for arm in self.arm_counts:
            if self.arm_rewards[arm]:
                stats[arm] = {
                    'count': self.arm_counts[arm],
                    'avg_reward': np.mean(self.arm_rewards[arm]),
                    'std_reward': np.std(self.arm_rewards[arm]),
                    'success_rate': np.mean([r > 0.5 for r in self.arm_rewards[arm]])
                }
            else:
                stats[arm] = {'count': 0, 'avg_reward': 0.0, 'std_reward': 0.0, 'success_rate': 0.0}
        return stats


class IntelligentModuleSelector:
    """Sophisticated module selection using multi-criteria analysis.
    
    Replaces simple round-robin with intelligent selection considering:
    1. Improvement potential based on performance trends
    2. Performance bottleneck analysis
    3. Historical mutation success rates
    4. Multi-armed bandit for exploration-exploitation balance
    """
    
    def __init__(self):
        self.performance_tracker: Dict[str, List[float]] = defaultdict(list)
        self.mutation_success_tracker: Dict[str, List[bool]] = defaultdict(list)
        self.bandit_solver = MultiArmedBandit()
        self.selection_history: List[ModuleAnalysis] = []
        
        # Criterion weights (can be adaptive)
        self.criterion_weights = {
            SelectionCriterion.IMPROVEMENT_POTENTIAL: 0.3,
            SelectionCriterion.PERFORMANCE_BOTTLENECK: 0.3,
            SelectionCriterion.MUTATION_SUCCESS_RATE: 0.2,
            SelectionCriterion.EXPLORATION_EXPLOITATION: 0.2
        }
    
    def select_target_module(
        self,
        system: Any,  # CompoundAISystem type
        context: SelectionContext
    ) -> str:
        """Select module using intelligent multi-criteria approach."""
        
        module_ids = list(system.modules.keys())
        if not module_ids:
            raise ValueError("No modules available for selection")
        
        if len(module_ids) == 1:
            return module_ids[0]
        
        # Analyze all modules
        module_analyses = {}
        for module_id in module_ids:
            analysis = self._analyze_module(module_id, system, context)
            module_analyses[module_id] = analysis
        
        # Select best module based on composite score
        best_module = max(module_analyses.keys(), 
                         key=lambda mid: module_analyses[mid].overall_score)
        
        # Track selection for learning
        self.selection_history.append(module_analyses[best_module])
        
        # Update bandit (will be updated with actual reward later)
        self.bandit_solver.arm_counts[best_module] += 1
        self.bandit_solver.total_selections += 1
        
        logger.debug(f"Selected module {best_module} with score {module_analyses[best_module].overall_score:.3f}")
        
        return best_module
    
    def _analyze_module(
        self,
        module_id: str,
        system: Any,
        context: SelectionContext
    ) -> ModuleAnalysis:
        """Comprehensive analysis of a single module."""
        
        # Criterion 1: Improvement potential
        improvement_potential = self._calculate_improvement_potential(
            module_id, context.performance_history
        )
        
        # Criterion 2: Performance bottleneck analysis
        bottleneck_score = self._analyze_performance_bottleneck(
            module_id, system, context.performance_history
        )
        
        # Criterion 3: Mutation success rate
        mutation_success_rate = self._get_mutation_success_rate(
            module_id, context.mutation_history
        )
        
        # Criterion 4: Multi-armed bandit exploration-exploitation
        bandit_score = self._get_bandit_score(module_id, list(system.modules.keys()))
        
        # Combine criteria with current weights
        overall_score = (
            self.criterion_weights[SelectionCriterion.IMPROVEMENT_POTENTIAL] * improvement_potential +
            self.criterion_weights[SelectionCriterion.PERFORMANCE_BOTTLENECK] * bottleneck_score +
            self.criterion_weights[SelectionCriterion.MUTATION_SUCCESS_RATE] * mutation_success_rate +
            self.criterion_weights[SelectionCriterion.EXPLORATION_EXPLOITATION] * bandit_score
        )
        
        # Generate selection reasons
        reasons = self._generate_selection_reasons(
            module_id, improvement_potential, bottleneck_score, 
            mutation_success_rate, bandit_score
        )
        
        return ModuleAnalysis(
            module_id=module_id,
            improvement_potential=improvement_potential,
            bottleneck_score=bottleneck_score,
            mutation_success_rate=mutation_success_rate,
            bandit_score=bandit_score,
            overall_score=overall_score,
            selection_reasons=reasons
        )
    
    def _calculate_improvement_potential(
        self,
        module_id: str,
        performance_history: Dict[str, List[float]]
    ) -> float:
        """Calculate improvement potential using performance analysis."""
        
        if module_id not in performance_history or not performance_history[module_id]:
            return 0.5  # Default for new modules - moderate potential
        
        history = performance_history[module_id]
        if len(history) < 3:
            return 0.5  # Not enough data
        
        # Analyze performance trends
        recent_window = min(5, len(history))
        recent_performance = np.mean(history[-recent_window:])
        overall_performance = np.mean(history)
        performance_variance = np.var(history)
        
        # Calculate improvement potential factors:
        
        # 1. Trend factor: declining recent performance suggests room for improvement
        if len(history) >= 6:
            older_performance = np.mean(history[-recent_window*2:-recent_window])
            trend_factor = max(0, older_performance - recent_performance)
        else:
            trend_factor = max(0, overall_performance - recent_performance)
        
        # 2. Variance factor: high variance suggests unstable performance
        max_possible_variance = 0.25  # Assume scores in [0,1] range
        variance_factor = min(1.0, performance_variance / max_possible_variance)
        
        # 3. Ceiling factor: distance from theoretical maximum performance
        theoretical_max = 1.0  # Assuming normalized scores
        ceiling_factor = max(0, theoretical_max - recent_performance)
        
        # 4. Stagnation factor: long periods of similar performance
        if len(history) >= 10:
            last_10 = history[-10:]
            if np.std(last_10) < 0.05:  # Very stable performance
                stagnation_factor = 0.3  # Moderate improvement potential
            else:
                stagnation_factor = 0.0
        else:
            stagnation_factor = 0.0
        
        # Combine factors
        improvement_potential = (
            0.3 * trend_factor +
            0.2 * variance_factor + 
            0.4 * ceiling_factor +
            0.1 * stagnation_factor
        )
        
        return min(1.0, max(0.0, improvement_potential))
    
    def _analyze_performance_bottleneck(
        self,
        module_id: str,
        system: Any,
        performance_history: Dict[str, List[float]]
    ) -> float:
        """Identify if module is a performance bottleneck."""
        
        if not performance_history:
            return 0.3  # Default moderate bottleneck score
        
        try:
            # Get performance statistics for this module
            if module_id not in performance_history or not performance_history[module_id]:
                module_performance = 0.5  # Default
            else:
                module_performance = np.mean(performance_history[module_id][-5:])  # Recent average
            
            # Compare with other modules
            other_modules_performance = []
            for other_id, other_history in performance_history.items():
                if other_id != module_id and other_history:
                    other_performance = np.mean(other_history[-5:])
                    other_modules_performance.append(other_performance)
            
            if not other_modules_performance:
                return 0.5  # Can't compare, assume moderate
            
            avg_other_performance = np.mean(other_modules_performance)
            
            # Bottleneck factors:
            
            # 1. Relative performance: lower performance suggests bottleneck
            if avg_other_performance > 0:
                relative_performance = module_performance / avg_other_performance
                performance_factor = max(0, 2.0 - relative_performance)  # Higher score for lower performance
            else:
                performance_factor = 0.5
            
            # 2. Workflow position factor (modules later in flow are more critical)
            # This is simplified - in full implementation would analyze actual workflow
            module_list = list(system.modules.keys())
            if module_id in module_list:
                position_factor = module_list.index(module_id) / max(len(module_list) - 1, 1)
            else:
                position_factor = 0.5
            
            # 3. Error propagation factor (modules that fail often create bottlenecks)
            if module_id in performance_history:
                recent_scores = performance_history[module_id][-10:]
                if recent_scores:
                    failure_rate = len([s for s in recent_scores if s < 0.3]) / len(recent_scores)
                    error_factor = failure_rate
                else:
                    error_factor = 0.0
            else:
                error_factor = 0.0
            
            # Combine factors
            bottleneck_score = (
                0.5 * performance_factor +
                0.2 * position_factor +
                0.3 * error_factor
            )
            
            return min(1.0, max(0.0, bottleneck_score))
            
        except Exception as e:
            logger.debug(f"Bottleneck analysis failed for {module_id}: {e}")
            return 0.3
    
    def _get_mutation_success_rate(
        self,
        module_id: str,
        mutation_history: Dict[str, List[bool]]
    ) -> float:
        """Get historical mutation success rate for module."""
        
        if module_id not in mutation_history or not mutation_history[module_id]:
            return 0.5  # Default for modules with no history
        
        history = mutation_history[module_id]
        recent_history = history[-10:]  # Focus on recent mutations
        
        if not recent_history:
            return 0.5
        
        success_rate = sum(recent_history) / len(recent_history)
        
        # Boost score for modules that consistently succeed
        if len(recent_history) >= 5 and success_rate > 0.7:
            return min(1.0, success_rate + 0.1)
        
        # Penalize modules that consistently fail
        if len(recent_history) >= 5 and success_rate < 0.3:
            return max(0.0, success_rate - 0.1)
        
        return success_rate
    
    def _get_bandit_score(self, module_id: str, all_modules: List[str]) -> float:
        """Get multi-armed bandit score for exploration-exploitation."""
        
        # If this is the bandit's choice, give it high score
        bandit_choice = self.bandit_solver.select_arm(all_modules)
        if bandit_choice == module_id:
            return 1.0
        
        # Otherwise, score based on historical performance and exploration need
        stats = self.bandit_solver.get_arm_statistics()
        
        if module_id not in stats or stats[module_id]['count'] == 0:
            return 0.8  # High score for unexplored modules
        
        module_stats = stats[module_id]
        
        # Balance exploitation (high average reward) with exploration (uncertainty)
        avg_reward = module_stats['avg_reward']
        exploration_bonus = 1.0 / (1.0 + module_stats['count'])  # Higher bonus for less explored
        
        return 0.7 * avg_reward + 0.3 * exploration_bonus
    
    def _generate_selection_reasons(
        self,
        module_id: str,
        improvement_potential: float,
        bottleneck_score: float,
        mutation_success_rate: float,
        bandit_score: float
    ) -> List[str]:
        """Generate human-readable reasons for module selection."""
        
        reasons = []
        
        if improvement_potential > 0.7:
            reasons.append(f"High improvement potential ({improvement_potential:.2f})")
        elif improvement_potential < 0.3:
            reasons.append(f"Low improvement potential ({improvement_potential:.2f})")
        
        if bottleneck_score > 0.6:
            reasons.append(f"Performance bottleneck detected ({bottleneck_score:.2f})")
        
        if mutation_success_rate > 0.7:
            reasons.append(f"High mutation success rate ({mutation_success_rate:.2f})")
        elif mutation_success_rate < 0.3:
            reasons.append(f"Low mutation success rate ({mutation_success_rate:.2f})")
        
        if bandit_score > 0.8:
            reasons.append(f"Exploration opportunity ({bandit_score:.2f})")
        
        if not reasons:
            reasons.append("Balanced selection across criteria")
        
        return reasons
    
    def update_mutation_result(self, module_id: str, success: bool, performance_gain: float = 0.0) -> None:
        """Update tracking with mutation result."""
        
        # Update mutation success tracking
        self.mutation_success_tracker[module_id].append(success)
        
        # Keep history manageable
        if len(self.mutation_success_tracker[module_id]) > 50:
            self.mutation_success_tracker[module_id] = self.mutation_success_tracker[module_id][-50:]
        
        # Update bandit with reward (success + performance gain)
        reward = (0.5 if success else 0.0) + max(0, min(0.5, performance_gain))
        self.bandit_solver.update_reward(module_id, reward)
    
    def update_performance_history(self, module_id: str, performance_score: float) -> None:
        """Update performance tracking for module."""
        
        self.performance_tracker[module_id].append(performance_score)
        
        # Keep history manageable
        if len(self.performance_tracker[module_id]) > 100:
            self.performance_tracker[module_id] = self.performance_tracker[module_id][-100:]
    
    def adapt_selection_weights(self, feedback: Dict[SelectionCriterion, float]) -> None:
        """Adapt criterion weights based on feedback."""
        
        for criterion, adjustment in feedback.items():
            if criterion in self.criterion_weights:
                self.criterion_weights[criterion] = max(0.1, min(0.8, 
                    self.criterion_weights[criterion] + adjustment))
        
        # Renormalize weights
        total_weight = sum(self.criterion_weights.values())
        for criterion in self.criterion_weights:
            self.criterion_weights[criterion] /= total_weight
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get statistics about selection performance."""
        
        if not self.selection_history:
            return {"message": "No selection history available"}
        
        recent_selections = self.selection_history[-20:]
        
        return {
            "total_selections": len(self.selection_history),
            "recent_avg_scores": {
                "improvement_potential": np.mean([s.improvement_potential for s in recent_selections]),
                "bottleneck_score": np.mean([s.bottleneck_score for s in recent_selections]),
                "mutation_success_rate": np.mean([s.mutation_success_rate for s in recent_selections]),
                "bandit_score": np.mean([s.bandit_score for s in recent_selections])
            },
            "current_weights": dict(self.criterion_weights),
            "bandit_statistics": self.bandit_solver.get_arm_statistics()
        }