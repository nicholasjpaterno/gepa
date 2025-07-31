"""Bayesian Hyperparameter Optimization with Transfer Learning."""

import logging
import numpy as np
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime
import random

from .config import HyperOptConfig
from .state import OptimizationState

logger = logging.getLogger(__name__)


@dataclass
class OptimizationContext:
    """Context information for transfer learning."""
    algorithm: str
    problem_features: Dict[str, Any]
    performance_trajectory: List[float]
    dataset_size: int
    task_type: str
    timestamp: datetime
    
    def similarity_score(self, other: "OptimizationContext") -> float:
        """Calculate similarity between contexts."""
        score = 0.0
        
        # Algorithm match
        if self.algorithm == other.algorithm:
            score += 0.3
        
        # Task type match
        if self.task_type == other.task_type:
            score += 0.3
        
        # Dataset size similarity
        size_ratio = min(self.dataset_size, other.dataset_size) / max(self.dataset_size, other.dataset_size)
        score += 0.2 * size_ratio
        
        # Problem features similarity
        if self.problem_features and other.problem_features:
            feature_score = 0.0
            common_features = set(self.problem_features.keys()) & set(other.problem_features.keys())
            
            for feature in common_features:
                if isinstance(self.problem_features[feature], (int, float)) and \
                   isinstance(other.problem_features[feature], (int, float)):
                    # Numerical feature similarity
                    val1 = float(self.problem_features[feature])
                    val2 = float(other.problem_features[feature])
                    if max(val1, val2) > 0:
                        feature_score += min(val1, val2) / max(val1, val2)
                elif self.problem_features[feature] == other.problem_features[feature]:
                    # Categorical feature match
                    feature_score += 1.0
            
            if common_features:
                score += 0.2 * (feature_score / len(common_features))
        
        return score


@dataclass
class HyperparameterObservation:
    """Single hyperparameter observation."""
    hyperparams: Dict[str, Any]
    performance: float
    fidelity: str
    context: OptimizationContext
    cost: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hyperparams": self.hyperparams,
            "performance": self.performance,
            "fidelity": self.fidelity,
            "context": asdict(self.context),
            "cost": self.cost
        }


class GaussianProcessOptimizer:
    """Gaussian Process for Bayesian optimization."""
    
    def __init__(self, kernel: str = "matern52"):
        self.kernel = kernel
        self.observations = []
        self.X = []  # Hyperparameter vectors
        self.y = []  # Performance values
        self.fidelities = []  # Fidelity levels
        
        # GP hyperparameters (simplified)
        self.noise_variance = 0.01
        self.signal_variance = 1.0
        self.length_scale = 1.0
        
    def warm_start_from_transfer(self, similar_contexts: List[HyperparameterObservation]) -> None:
        """Initialize GP with transferred knowledge."""
        logger.info(f"Warm-starting GP with {len(similar_contexts)} transferred observations")
        
        for obs in similar_contexts:
            # Convert hyperparams to vector
            param_vector = self._hyperparams_to_vector(obs.hyperparams)
            
            # Weight by context similarity and recency
            weight = 0.8  # Reduced weight for transferred data
            
            self.X.append(param_vector)
            self.y.append(obs.performance * weight)
            self.fidelities.append(obs.fidelity)
    
    def suggest(
        self,
        acquisition_function: str = "expected_improvement",
        exploration_weight: float = 0.1
    ) -> Dict[str, Any]:
        """Suggest next hyperparameters using acquisition function."""
        
        if len(self.X) < 3:
            # Random suggestions for initial exploration
            return self._random_suggestion()
        
        # Find best hyperparameters using acquisition function
        best_params = self._optimize_acquisition(acquisition_function, exploration_weight)
        
        return best_params
    
    def _random_suggestion(self) -> Dict[str, Any]:
        """Generate random hyperparameter suggestion."""
        return {
            "learning_rate": random.uniform(0.001, 0.01),
            "batch_size": random.choice([16, 32, 64]),
            "temperature": random.uniform(0.1, 1.0),
            "rollouts": random.randint(1, 5),
            "improvement_factor": random.uniform(0.01, 0.05),
            "cost": random.uniform(0.005, 0.02)
        }
    
    def _optimize_acquisition(
        self,
        acquisition_function: str,
        exploration_weight: float
    ) -> Dict[str, Any]:
        """Optimize acquisition function to find next suggestion."""
        # Simplified acquisition optimization
        # In practice, would use gradient-based optimization
        
        best_score = -np.inf
        best_params = None
        
        # Sample candidate points
        for _ in range(100):
            candidate_params = self._random_suggestion()
            param_vector = self._hyperparams_to_vector(candidate_params)
            
            # Predict mean and variance
            mean, variance = self._predict(param_vector)
            
            # Compute acquisition score
            if acquisition_function == "expected_improvement":
                score = self._expected_improvement(mean, variance, exploration_weight)
            elif acquisition_function == "upper_confidence_bound":
                score = mean + exploration_weight * np.sqrt(variance)
            else:
                score = mean  # Greedy selection
            
            if score > best_score:
                best_score = score
                best_params = candidate_params
        
        return best_params or self._random_suggestion()
    
    def _expected_improvement(self, mean: float, variance: float, exploration_weight: float) -> float:
        """Compute expected improvement acquisition function."""
        if not self.y:
            return mean
        
        best_f = max(self.y)
        improvement = mean - best_f
        
        if variance <= 0:
            return max(0, improvement)
        
        std = np.sqrt(variance)
        z = improvement / std
        
        # Simplified EI computation
        ei = improvement * self._norm_cdf(z) + std * self._norm_pdf(z)
        
        return ei
    
    def _norm_cdf(self, x: float) -> float:
        """Approximation of standard normal CDF."""
        return 0.5 * (1 + np.tanh(x * np.sqrt(2 / np.pi)))
    
    def _norm_pdf(self, x: float) -> float:
        """Standard normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def _predict(self, param_vector: np.ndarray) -> Tuple[float, float]:
        """Predict mean and variance at given hyperparameters."""
        if not self.X:
            return 0.0, 1.0
        
        # Simplified GP prediction using kernel similarity
        similarities = []
        for x in self.X:
            similarity = self._kernel_similarity(param_vector, np.array(x))
            similarities.append(similarity)
        
        similarities = np.array(similarities)
        weights = similarities / (np.sum(similarities) + 1e-8)
        
        # Weighted mean prediction
        mean = np.sum(weights * np.array(self.y))
        
        # Simplified variance (higher when farther from observed points)
        max_similarity = np.max(similarities) if similarities.size > 0 else 0
        variance = self.signal_variance * (1 - max_similarity + self.noise_variance)
        
        return mean, variance
    
    def _kernel_similarity(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute kernel similarity between parameter vectors."""
        distance = np.linalg.norm(x1 - x2)
        
        if self.kernel == "rbf":
            return self.signal_variance * np.exp(-0.5 * (distance / self.length_scale)**2)
        elif self.kernel == "matern52":
            # Simplified Matern 5/2 kernel
            scaled_dist = np.sqrt(5) * distance / self.length_scale
            return self.signal_variance * (1 + scaled_dist + scaled_dist**2 / 3) * np.exp(-scaled_dist)
        else:
            # Linear kernel
            return self.signal_variance * max(0, 1 - distance / self.length_scale)
    
    def _hyperparams_to_vector(self, hyperparams: Dict[str, Any]) -> np.ndarray:
        """Convert hyperparameters to numerical vector."""
        # Fixed order for consistency
        keys = ["learning_rate", "batch_size", "temperature", "rollouts", "improvement_factor", "cost"]
        
        vector = []
        for key in keys:
            if key in hyperparams:
                if key == "batch_size":
                    # Map batch sizes to normalized values
                    batch_map = {16: 0.0, 32: 0.5, 64: 1.0}
                    vector.append(batch_map.get(hyperparams[key], 0.5))
                else:
                    vector.append(float(hyperparams[key]))
            else:
                vector.append(0.0)  # Default value
        
        return np.array(vector)
    
    def update(self, hyperparams: Dict[str, Any], performance: float, fidelity: str) -> None:
        """Update GP with new observation."""
        param_vector = self._hyperparams_to_vector(hyperparams)
        
        self.X.append(param_vector.tolist())
        self.y.append(performance)
        self.fidelities.append(fidelity)
        
        # Limit history to prevent memory issues
        if len(self.X) > 1000:
            self.X = self.X[-500:]
            self.y = self.y[-500:]
            self.fidelities = self.fidelities[-500:]
    
    def get_uncertainty(self, hyperparams: Dict[str, Any]) -> float:
        """Get prediction uncertainty for given hyperparameters."""
        param_vector = self._hyperparams_to_vector(hyperparams)
        _, variance = self._predict(param_vector)
        return np.sqrt(variance)


class HyperparameterTransferEngine:
    """Manages transfer learning between optimization runs."""
    
    def __init__(self, max_stored_contexts: int = 1000):
        self.stored_contexts = []
        self.max_stored_contexts = max_stored_contexts
    
    def find_similar_contexts(
        self,
        algorithm: str,
        problem_characteristics: Dict[str, Any],
        performance_history: List[float],
        similarity_threshold: float = 0.7,
        max_similar: int = 10
    ) -> List[HyperparameterObservation]:
        """Find similar optimization contexts for transfer learning."""
        
        current_context = OptimizationContext(
            algorithm=algorithm,
            problem_features=problem_characteristics,
            performance_trajectory=performance_history,
            dataset_size=problem_characteristics.get("data_size", 0),
            task_type=problem_characteristics.get("task_type", "general"),
            timestamp=datetime.now()
        )
        
        similar_observations = []
        
        for obs in self.stored_contexts:
            similarity = current_context.similarity_score(obs.context)
            
            if similarity >= similarity_threshold:
                similar_observations.append((obs, similarity))
        
        # Sort by similarity and return top matches
        similar_observations.sort(key=lambda x: x[1], reverse=True)
        
        return [obs for obs, _ in similar_observations[:max_similar]]
    
    def store_experience(
        self,
        hyperparams: Dict[str, Any],
        performance: float,
        context: OptimizationContext
    ) -> None:
        """Store optimization experience for future transfer."""
        observation = HyperparameterObservation(
            hyperparams=hyperparams,
            performance=performance,
            fidelity="high",  # Default fidelity
            context=context
        )
        
        self.stored_contexts.append(observation)
        
        # Limit storage
        if len(self.stored_contexts) > self.max_stored_contexts:
            self.stored_contexts = self.stored_contexts[-self.max_stored_contexts//2:]


class MultiFidelityManager:
    """Manages multi-fidelity optimization."""
    
    def __init__(self):
        self.fidelity_costs = {
            "low": 0.1,
            "medium": 0.5,
            "high": 1.0
        }
        self.fidelity_accuracy = {
            "low": 0.7,
            "medium": 0.85,
            "high": 1.0
        }
    
    def select_fidelity(
        self,
        budget_remaining: int,
        uncertainty: float,
        exploration_phase: bool = True
    ) -> str:
        """Select appropriate fidelity level based on budget and uncertainty."""
        
        # Early exploration: use low fidelity
        if exploration_phase and budget_remaining > 50:
            return "low"
        
        # High uncertainty: start with lower fidelity
        if uncertainty > 0.5:
            return "medium" if budget_remaining > 20 else "low"
        
        # Low uncertainty or limited budget: use high fidelity
        if budget_remaining > 10:
            return "high"
        elif budget_remaining > 5:
            return "medium"
        else:
            return "low"
    
    def adjust_performance(self, performance: float, fidelity: str) -> float:
        """Adjust performance score based on fidelity."""
        accuracy = self.fidelity_accuracy[fidelity]
        return performance * accuracy


class BayesianHyperOptimizer:
    """
    Advanced hyperparameter optimization with transfer learning.
    
    Combines Bayesian optimization with transfer learning and multi-fidelity
    evaluation for unprecedented efficiency in hyperparameter tuning.
    """
    
    def __init__(self, config: HyperOptConfig):
        self.config = config
        
        # Gaussian Process for Bayesian optimization
        self.gp_optimizer = GaussianProcessOptimizer(kernel=config.gp_kernel)
        
        # Transfer learning from previous optimizations
        self.transfer_engine = HyperparameterTransferEngine()
        
        # Multi-fidelity optimization
        self.fidelity_manager = MultiFidelityManager()
        
        # Current optimization state
        self.current_context = None
        self.optimization_history = []
        
        logger.info("BayesianHyperOptimizer initialized with transfer learning and multi-fidelity")
    
    async def suggest_hyperparameters(
        self,
        algorithm_choice: str,
        current_state: OptimizationState
    ) -> Tuple[Dict[str, Any], str]:
        """
        Suggest optimal hyperparameters using Bayesian optimization + transfer learning.
        """
        # Create current context
        self.current_context = OptimizationContext(
            algorithm=algorithm_choice,
            problem_features=current_state.problem_features.__dict__,
            performance_trajectory=current_state.performance_trajectory,
            dataset_size=len(current_state.dataset),
            task_type=current_state.problem_features.task_type,
            timestamp=datetime.now()
        )
        
        # Retrieve similar past optimizations
        similar_contexts = []
        if self.config.transfer_learning_enabled:
            similar_contexts = self.transfer_engine.find_similar_contexts(
                algorithm=algorithm_choice,
                problem_characteristics=self.current_context.problem_features,
                performance_history=current_state.performance_trajectory,
                similarity_threshold=self.config.similarity_threshold,
                max_similar=self.config.max_similar_contexts
            )
            
            logger.debug(f"Found {len(similar_contexts)} similar contexts for transfer learning")
        
        # Initialize GP with transferred knowledge
        if similar_contexts:
            self.gp_optimizer.warm_start_from_transfer(similar_contexts)
        
        # Bayesian optimization with acquisition function
        suggested_params = self.gp_optimizer.suggest(
            acquisition_function=self.config.acquisition_function,
            exploration_weight=self._adaptive_exploration_weight(current_state)
        )
        
        # Multi-fidelity evaluation selection
        fidelity_level = "high"  # Default
        if self.config.multi_fidelity_enabled:
            uncertainty = self.gp_optimizer.get_uncertainty(suggested_params)
            fidelity_level = self.fidelity_manager.select_fidelity(
                budget_remaining=current_state.budget_remaining,
                uncertainty=uncertainty,
                exploration_phase=len(self.optimization_history) < 10
            )
        
        logger.debug(
            f"Suggested hyperparams for {algorithm_choice}: {suggested_params} "
            f"(fidelity: {fidelity_level})"
        )
        
        return suggested_params, fidelity_level
    
    def _adaptive_exploration_weight(self, current_state: OptimizationState) -> float:
        """Compute adaptive exploration weight based on optimization progress."""
        base_weight = self.config.exploration_weight
        
        # Increase exploration if performance is stagnating
        stagnation_counter = current_state.performance_metrics.convergence_signals.get(
            "stagnation_counter", 0
        )
        stagnation_bonus = min(0.2, stagnation_counter * 0.02)
        
        # Decrease exploration as budget diminishes
        budget_factor = current_state.budget_remaining / current_state.initial_budget
        budget_reduction = (1 - budget_factor) * 0.1
        
        adaptive_weight = base_weight + stagnation_bonus - budget_reduction
        return max(0.05, min(0.5, adaptive_weight))
    
    def update_model(
        self,
        hyperparams: Dict[str, Any],
        performance: float,
        fidelity_level: str
    ) -> None:
        """Update Bayesian model with new observations."""
        
        # Adjust performance based on fidelity
        if self.config.multi_fidelity_enabled:
            adjusted_performance = self.fidelity_manager.adjust_performance(
                performance, fidelity_level
            )
        else:
            adjusted_performance = performance
        
        # Multi-fidelity GP update
        self.gp_optimizer.update(hyperparams, adjusted_performance, fidelity_level)
        
        # Store for future transfer learning
        if self.current_context:
            self.transfer_engine.store_experience(
                hyperparams, adjusted_performance, self.current_context
            )
        
        # Track optimization history
        self.optimization_history.append({
            "hyperparams": hyperparams,
            "performance": performance,
            "adjusted_performance": adjusted_performance,
            "fidelity": fidelity_level,
            "timestamp": datetime.now()
        })
        
        logger.debug(
            f"Updated hyperparameter model: performance {performance:.3f} "
            f"(adjusted: {adjusted_performance:.3f}, fidelity: {fidelity_level})"
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive hyperparameter optimization metrics."""
        
        # Performance statistics
        if self.optimization_history:
            performances = [h["adjusted_performance"] for h in self.optimization_history]
            performance_stats = {
                "mean_performance": np.mean(performances),
                "max_performance": np.max(performances),
                "performance_improvement": performances[-1] - performances[0] if len(performances) > 1 else 0.0,
                "performance_std": np.std(performances)
            }
        else:
            performance_stats = {
                "mean_performance": 0.0,
                "max_performance": 0.0,
                "performance_improvement": 0.0,
                "performance_std": 0.0
            }
        
        # Fidelity usage
        fidelity_usage = defaultdict(int)
        for h in self.optimization_history:
            fidelity_usage[h["fidelity"]] += 1
        
        return {
            "optimization_rounds": len(self.optimization_history),
            "transfer_contexts_stored": len(self.transfer_engine.stored_contexts),
            "gp_observations": len(self.gp_optimizer.y),
            "performance_stats": performance_stats,
            "fidelity_usage": dict(fidelity_usage),
            "recent_hyperparams": [
                h["hyperparams"] for h in self.optimization_history[-5:]
            ] if self.optimization_history else []
        }