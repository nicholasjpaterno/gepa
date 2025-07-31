"""Optimization state management for MetaOrchestrator."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime

from ..core.system import CompoundAISystem


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization tracking."""
    current_best: float = 0.0
    recent_improvements: List[float] = field(default_factory=list)
    improvement_velocity: float = 0.0
    convergence_signals: Dict[str, float] = field(default_factory=dict)
    exploration_metrics: Dict[str, float] = field(default_factory=dict)
    
    def update(self, new_score: float) -> None:
        """Update metrics with new performance score."""
        if new_score > self.current_best:
            improvement = new_score - self.current_best
            self.recent_improvements.append(improvement)
            self.current_best = new_score
            
            # Keep only recent improvements (last 10)
            if len(self.recent_improvements) > 10:
                self.recent_improvements = self.recent_improvements[-10:]
            
            # Calculate improvement velocity
            if len(self.recent_improvements) >= 2:
                self.improvement_velocity = np.mean(self.recent_improvements[-5:])
        
        # Update convergence signals
        self.convergence_signals["stagnation_counter"] = (
            self.convergence_signals.get("stagnation_counter", 0) + 1
            if new_score <= self.current_best
            else 0
        )


@dataclass
class ProblemFeatures:
    """Problem characteristics for optimization state."""
    domain: str = "unknown"
    complexity_score: float = 0.5
    data_size: int = 0
    input_dimensionality: int = 0
    output_complexity: float = 0.5
    task_type: str = "general"
    
    @classmethod
    def from_dataset(cls, dataset: List[Dict[str, Any]]) -> "ProblemFeatures":
        """Extract problem features from dataset."""
        features = cls()
        features.data_size = len(dataset)
        
        if dataset:
            # Analyze first few examples
            sample = dataset[0]
            if "text" in sample:
                features.input_dimensionality = len(str(sample["text"]).split())
            if "expected" in sample:
                features.output_complexity = len(str(sample["expected"]).split()) / 100.0
        
        # Infer task type from dataset structure
        if any("code" in str(item).lower() for item in dataset[:3]):
            features.task_type = "code_generation"
        elif any("summary" in str(item).lower() for item in dataset[:3]):
            features.task_type = "summarization"
        elif any("question" in str(item).lower() for item in dataset[:3]):
            features.task_type = "qa"
        
        return features


class OptimizationState:
    """Comprehensive optimization state for MetaOrchestrator."""
    
    def __init__(
        self,
        system: CompoundAISystem,
        dataset: List[Dict[str, Any]],
        budget: int
    ):
        self.system = system
        self.dataset = dataset
        self.initial_budget = budget
        self.budget_remaining = budget
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        self.problem_features = ProblemFeatures.from_dataset(dataset)
        
        # Algorithm tracking
        self.algorithm_history: List[Dict[str, Any]] = []
        self.parameter_history: List[Dict[str, Any]] = []
        self.topology_changes: List[Dict[str, Any]] = []
        self.prompt_changes: List[Dict[str, Any]] = []
        
        # Performance feedback
        self.recent_performance_feedback: List[float] = []
        self.performance_trajectory: List[float] = []
        
        # Timing
        self.start_time = datetime.now()
        self.generation = 0
        
    def update(self, algorithm_result: Dict[str, Any]) -> None:
        """Update state with algorithm result."""
        self.generation += 1
        self.budget_remaining -= algorithm_result.get("rollouts_used", 1)
        
        # Update performance
        if "score" in algorithm_result:
            score = algorithm_result["score"]
            self.performance_metrics.update(score)
            self.performance_trajectory.append(score)
            self.recent_performance_feedback.append(score)
            
            # Keep recent feedback limited
            if len(self.recent_performance_feedback) > 20:
                self.recent_performance_feedback = self.recent_performance_feedback[-20:]
        
        # Track algorithm usage
        if "algorithm" in algorithm_result:
            self.algorithm_history.append({
                "generation": self.generation,
                "algorithm": algorithm_result["algorithm"],
                "score": algorithm_result.get("score", 0.0),
                "cost": algorithm_result.get("cost", 0.0),
                "timestamp": datetime.now()
            })
        
        # Track parameter changes
        if "hyperparams" in algorithm_result:
            self.parameter_history.append({
                "generation": self.generation,
                "hyperparams": algorithm_result["hyperparams"],
                "performance": algorithm_result.get("score", 0.0)
            })
        
        # Track topology changes
        if "topology_change" in algorithm_result:
            self.topology_changes.append({
                "generation": self.generation,
                "change": algorithm_result["topology_change"],
                "performance_delta": algorithm_result.get("performance_delta", 0.0)
            })
        
        # Track prompt changes
        if "prompt_changes" in algorithm_result:
            self.prompt_changes.append({
                "generation": self.generation,
                "changes": algorithm_result["prompt_changes"],
                "performance_feedback": algorithm_result.get("score", 0.0)
            })
    
    def update_system(self, new_system: CompoundAISystem) -> None:
        """Update the current system."""
        self.system = new_system
    
    def encode(self) -> np.ndarray:
        """Encode current state for RL algorithm selector."""
        features = [
            self.performance_metrics.current_best,
            self.performance_metrics.improvement_velocity,
            float(self.budget_remaining) / self.initial_budget,
            self.problem_features.complexity_score,
            float(self.problem_features.data_size) / 1000.0,  # Normalized
            self.problem_features.output_complexity,
            len(self.performance_metrics.recent_improvements),
            self.performance_metrics.convergence_signals.get("stagnation_counter", 0) / 10.0,
            float(self.generation) / 100.0,  # Normalized generation
        ]
        
        # Add task type encoding
        task_encoding = {
            "general": [1, 0, 0, 0],
            "code_generation": [0, 1, 0, 0],
            "summarization": [0, 0, 1, 0],
            "qa": [0, 0, 0, 1]
        }
        features.extend(task_encoding.get(self.problem_features.task_type, [1, 0, 0, 0]))
        
        return np.array(features, dtype=np.float32)
    
    def next_encode(self) -> np.ndarray:
        """Encode next state (after potential update)."""
        # For now, return current encoding
        # In practice, this would be the state after applying an action
        return self.encode()
    
    def get_best_result(self) -> Dict[str, Any]:
        """Get the best optimization result."""
        # Calculate hyperopt improvements
        hyperopt_improvements = 0.0
        if len(self.parameter_history) > 1:
            initial_performance = self.parameter_history[0]["performance"]
            final_performance = self.parameter_history[-1]["performance"]
            if initial_performance > 0:
                hyperopt_improvements = (final_performance - initial_performance) / initial_performance
        
        return {
            "best_system": self.system,
            "best_score": self.performance_metrics.current_best,
            "total_rollouts": self.initial_budget - self.budget_remaining,
            "generations": self.generation,
            "optimization_time": (datetime.now() - self.start_time).total_seconds(),
            "performance_trajectory": self.performance_trajectory,
            "algorithm_history": self.algorithm_history,
            "topology_changes": len(self.topology_changes),
            "prompt_changes": len(self.prompt_changes),
            "hyperopt_gains": hyperopt_improvements
        }
    
    @property
    def should_continue(self) -> bool:
        """Check if optimization should continue."""
        return (
            self.budget_remaining > 0 and
            self.performance_metrics.convergence_signals.get("stagnation_counter", 0) < 20
        )