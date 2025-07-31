"""RL-based Algorithm Selector for MetaOrchestrator."""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import random

from .config import RLConfig
from .state import OptimizationState

logger = logging.getLogger(__name__)


class ExperienceReplay:
    """Experience replay buffer for RL training."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, experience: Tuple[np.ndarray, int, float, np.ndarray]) -> None:
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, int, float, np.ndarray]]:
        """Sample batch of experiences."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)


class OptimizationStateEncoder:
    """Encodes optimization state into features for RL agent."""
    
    def __init__(self):
        self.dim = 13  # Based on OptimizationState.encode()
    
    def encode(
        self,
        current_performance: float,
        improvement_velocity: float,
        budget_remaining: float,
        problem_characteristics: Dict[str, Any],
        search_space_exploration: Dict[str, float],
        convergence_indicators: Dict[str, float]
    ) -> np.ndarray:
        """Encode optimization state into feature vector."""
        features = [
            current_performance,
            improvement_velocity,
            budget_remaining,
            problem_characteristics.get("complexity_score", 0.5),
            problem_characteristics.get("data_size", 0) / 1000.0,
            problem_characteristics.get("output_complexity", 0.5),
            len(search_space_exploration),
            convergence_indicators.get("stagnation_counter", 0) / 10.0,
            problem_characteristics.get("generation", 0) / 100.0,
        ]
        
        # Add task type encoding (one-hot)
        task_type = problem_characteristics.get("task_type", "general")
        task_encoding = {
            "general": [1, 0, 0, 0],
            "code_generation": [0, 1, 0, 0],
            "summarization": [0, 0, 1, 0],
            "qa": [0, 0, 0, 1]
        }
        features.extend(task_encoding.get(task_type, [1, 0, 0, 0]))
        
        return np.array(features, dtype=np.float32)


class ActorCriticNetwork:
    """Actor-Critic network for algorithm selection."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 128]
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # Simple implementation - in practice would use PyTorch/TensorFlow
        self.actor_weights = np.random.randn(state_dim, action_dim) * 0.1
        self.critic_weights = np.random.randn(state_dim, 1) * 0.1
        
        logger.info(f"ActorCritic network initialized: {state_dim} -> {hidden_dims} -> {action_dim}")
    
    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward pass through network."""
        # Simple linear transformation (placeholder for real neural network)
        action_logits = np.dot(state, self.actor_weights)
        action_probs = self._softmax(action_logits)
        
        value_estimate = np.dot(state, self.critic_weights).item()
        
        return action_probs, value_estimate
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Softmax activation with numerical stability."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def update_weights(self, loss: float) -> None:
        """Update network weights (placeholder)."""
        # In practice, this would use gradient descent
        learning_rate = 0.001
        self.actor_weights *= (1 - learning_rate * loss)
        self.critic_weights *= (1 - learning_rate * loss)


class AdaptiveRewardShaper:
    """Shapes rewards for RL training."""
    
    def __init__(self):
        self.baseline_performance = 0.0
        self.performance_history = deque(maxlen=100)
    
    def shape_rewards(self, trajectory: List[Dict[str, Any]]) -> List[float]:
        """Shape rewards based on performance improvements and efficiency."""
        rewards = []
        
        for step in trajectory:
            # Base reward from performance improvement
            improvement = step.get("improvement", 0.0)
            base_reward = improvement * 10.0  # Scale improvement
            
            # Efficiency bonus (reward algorithms that use budget wisely)
            rollouts_used = step.get("rollouts_used", 1)
            efficiency_bonus = max(0, (5 - rollouts_used) * 0.1)
            
            # Convergence penalty (penalize stagnation)
            stagnation_penalty = -step.get("stagnation_counter", 0) * 0.01
            
            # Exploration bonus (reward trying different algorithms)
            algorithm_diversity = step.get("algorithm_diversity", 0.5)
            exploration_bonus = algorithm_diversity * 0.05
            
            total_reward = base_reward + efficiency_bonus + stagnation_penalty + exploration_bonus
            rewards.append(total_reward)
            
            # Update performance tracking
            if improvement > 0:
                self.performance_history.append(step.get("score", 0.0))
                if len(self.performance_history) > 10:
                    self.baseline_performance = np.mean(list(self.performance_history)[-10:])
        
        return rewards


class RLAlgorithmSelector:
    """
    Reinforcement Learning agent that learns optimal algorithm sequencing.
    
    Uses Actor-Critic with experience replay to learn when to apply
    which optimization algorithm based on current optimization state.
    """
    
    def __init__(self, config: RLConfig):
        self.config = config
        
        # State representation
        self.state_encoder = OptimizationStateEncoder()
        
        # RL agent (Actor-Critic network)
        self.policy_network = ActorCriticNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims
        )
        
        # Experience replay and learning
        self.experience_buffer = ExperienceReplay(capacity=config.buffer_capacity)
        self.reward_shaper = AdaptiveRewardShaper()
        
        # Training state
        self.training_mode = True
        self.warmup_completed = False
        self.update_counter = 0
        
        # Available algorithms (must match action_dim)
        self.available_algorithms = [
            "reflective_mutation",
            "pareto_sampling", 
            "system_aware_merge",
            "structural_mutation",
            "crossover",
            "random_search"
        ]
        
        # Performance tracking
        self.selection_history = deque(maxlen=1000)
        self.performance_by_algorithm = {alg: [] for alg in self.available_algorithms}
        
        logger.info("RLAlgorithmSelector initialized with Actor-Critic architecture")
    
    async def select_algorithm(
        self,
        optimization_state: OptimizationState,
        available_budget: int
    ) -> Tuple[str, float]:
        """
        Select optimal algorithm based on current state and constraints.
        """
        # Encode current optimization state
        state_features = optimization_state.encode()
        
        # Policy network selects algorithm + value estimate
        algorithm_probs, value_estimate = self.policy_network.forward(state_features)
        
        # Sample with exploration during training
        if self.training_mode and not self._should_exploit():
            action_idx = self._sample_with_exploration(algorithm_probs)
        else:
            action_idx = np.argmax(algorithm_probs)
        
        selected_algorithm = self.available_algorithms[action_idx]
        
        # Track selection
        self.selection_history.append({
            "algorithm": selected_algorithm,
            "state": state_features,
            "value_estimate": value_estimate,
            "exploration": self.training_mode and not self._should_exploit()
        })
        
        logger.debug(
            f"Selected algorithm: {selected_algorithm} "
            f"(value: {value_estimate:.3f}, probs: {algorithm_probs})"
        )
        
        return selected_algorithm, value_estimate
    
    def _should_exploit(self) -> bool:
        """Determine if we should exploit (vs explore) based on training progress."""
        if not self.warmup_completed:
            return len(self.experience_buffer) >= self.config.warmup_steps
        
        # Epsilon-greedy with decay
        epsilon = max(0.1, 1.0 - len(self.experience_buffer) / (self.config.buffer_capacity * 0.5))
        return random.random() > epsilon
    
    def _sample_with_exploration(self, algorithm_probs: np.ndarray) -> int:
        """Sample action with exploration noise."""
        # Add exploration noise
        noise = np.random.normal(0, self.config.exploration_noise, size=algorithm_probs.shape)
        noisy_probs = algorithm_probs + noise
        noisy_probs = np.exp(noisy_probs) / np.sum(np.exp(noisy_probs))  # Re-normalize
        
        # Sample from noisy distribution
        return np.random.choice(len(algorithm_probs), p=noisy_probs)
    
    def store_experience(self, experience: Tuple[np.ndarray, int, float, np.ndarray]) -> None:
        """Store experience for training."""
        self.experience_buffer.push(experience)
        
        # Update performance tracking
        state, action, reward, next_state = experience
        algorithm = self.available_algorithms[action]
        self.performance_by_algorithm[algorithm].append(reward)
        
        # Trigger training if enough experiences
        if len(self.experience_buffer) >= self.config.batch_size:
            if self.update_counter % self.config.update_frequency == 0:
                self._update_policy()
            self.update_counter += 1
        
        logger.debug(f"Stored experience: {algorithm} -> reward {reward:.3f}")
    
    def _update_policy(self) -> None:
        """Update policy based on experience replay (simplified PPO-style)."""
        if len(self.experience_buffer) < self.config.batch_size:
            return
        
        # Sample batch of experiences
        batch = self.experience_buffer.sample(self.config.batch_size)
        
        # Compute losses (simplified version)
        total_loss = 0.0
        for state, action, reward, next_state in batch:
            # Compute TD error
            _, current_value = self.policy_network.forward(state)
            _, next_value = self.policy_network.forward(next_state)
            
            td_target = reward + self.config.gamma * next_value
            td_error = td_target - current_value
            
            total_loss += td_error ** 2
        
        avg_loss = total_loss / len(batch)
        
        # Update network weights
        self.policy_network.update_weights(avg_loss)
        
        if not self.warmup_completed and len(self.experience_buffer) >= self.config.warmup_steps:
            self.warmup_completed = True
            logger.info("RL warmup completed, switching to exploitation-focused learning")
        
        logger.debug(f"Policy updated: avg_loss={avg_loss:.4f}, buffer_size={len(self.experience_buffer)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive RL metrics."""
        algorithm_performance = {}
        for alg, rewards in self.performance_by_algorithm.items():
            if rewards:
                algorithm_performance[alg] = {
                    "mean_reward": np.mean(rewards),
                    "std_reward": np.std(rewards),
                    "count": len(rewards),
                    "recent_performance": np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
                }
            else:
                algorithm_performance[alg] = {
                    "mean_reward": 0.0,
                    "std_reward": 0.0,
                    "count": 0,
                    "recent_performance": 0.0
                }
        
        return {
            "buffer_size": len(self.experience_buffer),
            "training_mode": self.training_mode,
            "warmup_completed": self.warmup_completed,
            "update_counter": self.update_counter,
            "algorithm_performance": algorithm_performance,
            "recent_selections": [h["algorithm"] for h in list(self.selection_history)[-10:]],
            "baseline_performance": self.reward_shaper.baseline_performance
        }
    
    def set_training_mode(self, training: bool) -> None:
        """Set training mode."""
        self.training_mode = training
        logger.info(f"RLAlgorithmSelector training mode: {training}")
    
    def save_model(self, path: str) -> None:
        """Save model state (placeholder)."""
        # In practice, would serialize neural network weights
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model state (placeholder)."""
        # In practice, would load neural network weights
        logger.info(f"Model loaded from {path}")