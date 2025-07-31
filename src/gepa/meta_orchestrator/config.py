"""Enhanced Configuration System for MetaOrchestrator."""

import os
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Literal
from pydantic import BaseModel, Field, validator, model_validator
from enum import Enum

class OptimizationMode(str, Enum):
    """Optimization modes for different use cases."""
    EXPLORATION = "exploration"      # Focus on discovering new solutions
    EXPLOITATION = "exploitation"    # Focus on improving current best
    BALANCED = "balanced"           # Balance between exploration and exploitation
    CONSERVATIVE = "conservative"   # Minimal resource usage, safe choices
    AGGRESSIVE = "aggressive"       # Maximum resource usage, high risk/reward

class BudgetStrategy(str, Enum):
    """Budget allocation strategies."""
    FIXED = "fixed"                 # Equal allocation across components
    DYNAMIC = "dynamic"             # Allocation based on performance
    ADAPTIVE = "adaptive"           # Learning-based allocation
    PRIORITY = "priority"           # Priority-based allocation


class RLConfig(BaseModel):
    """Enhanced configuration for RL Algorithm Selector."""
    state_dim: int = Field(13, ge=1, le=100, description="Dimension of state representation")
    action_dim: int = Field(6, ge=2, le=20, description="Number of available algorithms")
    hidden_dims: List[int] = Field([256, 256, 128], description="Hidden layer dimensions")
    learning_rate: float = Field(3e-4, gt=0, lt=1, description="Learning rate for policy network")
    buffer_capacity: int = Field(10000, ge=1000, le=100000, description="Experience replay buffer size")
    batch_size: int = Field(64, ge=16, le=512, description="Training batch size")
    gamma: float = Field(0.99, ge=0.9, le=0.999, description="Discount factor")
    tau: float = Field(0.005, gt=0, lt=1, description="Soft update parameter")
    exploration_noise: float = Field(0.1, ge=0, le=1, description="Exploration noise level")
    warmup_steps: int = Field(1000, ge=100, le=10000, description="Steps before training")
    update_frequency: int = Field(4, ge=1, le=100, description="Steps between updates")
    
    # Advanced settings
    prioritized_replay: bool = Field(True, description="Use prioritized experience replay")
    double_q_learning: bool = Field(True, description="Use double Q-learning")
    dueling_network: bool = Field(False, description="Use dueling network architecture")
    
    @validator('hidden_dims')
    def validate_hidden_dims(cls, v):
        if not v or len(v) > 5:
            raise ValueError("Hidden dimensions must be 1-5 layers")
        if any(dim < 32 or dim > 1024 for dim in v):
            raise ValueError("Hidden dimensions must be between 32 and 1024")
        return v


class TopologyConfig(BaseModel):
    """Enhanced configuration for NEAT System Evolver."""
    max_complexity_threshold: float = Field(2.0, ge=1.0, le=10.0, description="Maximum system complexity")
    min_topology_budget: int = Field(10, ge=5, le=100, description="Minimum budget for topology evolution")
    min_improvement_threshold: float = Field(0.05, ge=0.01, le=0.5, description="Minimum improvement to evolve")
    mutation_types: List[str] = Field(
        ["add_module", "remove_module", "reconnect", "control_flow"],
        description="Available mutation types"
    )
    complexity_penalty: float = Field(0.1, ge=0, le=1, description="Penalty for system complexity")
    performance_prediction_enabled: bool = Field(True, description="Enable performance prediction")
    diversity_maintenance: bool = Field(True, description="Maintain topology diversity")
    
    # Advanced settings
    elitism_rate: float = Field(0.1, ge=0, le=0.5, description="Rate of elite individuals to preserve")
    mutation_rate: float = Field(0.3, ge=0.1, le=1.0, description="Probability of mutation")
    crossover_rate: float = Field(0.7, ge=0, le=1.0, description="Probability of crossover")
    
    @validator('mutation_types')
    def validate_mutation_types(cls, v):
        valid_types = {"add_module", "remove_module", "reconnect", "control_flow", "parameter_mutation"}
        if not all(mt in valid_types for mt in v):
            raise ValueError(f"Invalid mutation types. Valid: {valid_types}")
        return v


class HyperOptConfig(BaseModel):
    """Enhanced configuration for Bayesian HyperOptimizer."""
    acquisition_function: str = Field(
        "expected_improvement", 
        description="Acquisition function for Bayesian optimization"
    )
    n_initial_points: int = Field(5, ge=3, le=20, description="Initial random points")
    n_suggestions: int = Field(1, ge=1, le=10, description="Number of suggestions per iteration")
    transfer_learning_enabled: bool = Field(True, description="Enable transfer learning")
    multi_fidelity_enabled: bool = Field(True, description="Enable multi-fidelity optimization")
    max_similar_contexts: int = Field(10, ge=5, le=50, description="Max similar contexts for transfer")
    similarity_threshold: float = Field(0.7, ge=0.5, le=0.95, description="Similarity threshold")
    gp_kernel: str = Field("matern52", description="Gaussian Process kernel")
    exploration_weight: float = Field(0.1, ge=0, le=1, description="Exploration vs exploitation weight")
    
    # Advanced settings
    noise_level: float = Field(1e-6, gt=0, lt=1, description="GP noise level")
    normalize_inputs: bool = Field(True, description="Normalize input features")
    optimize_hyperparameters: bool = Field(True, description="Optimize GP hyperparameters")
    
    @validator('acquisition_function')
    def validate_acquisition_function(cls, v):
        valid_functions = {"expected_improvement", "probability_improvement", "upper_confidence_bound"}
        if v not in valid_functions:
            raise ValueError(f"Invalid acquisition function. Valid: {valid_functions}")
        return v
    
    @validator('gp_kernel')
    def validate_gp_kernel(cls, v):
        valid_kernels = {"matern52", "matern32", "rbf", "linear"}
        if v not in valid_kernels:
            raise ValueError(f"Invalid GP kernel. Valid: {valid_kernels}")
        return v


class PromptConfig(BaseModel):
    """Enhanced configuration for Prompt Structure Evolver."""
    grammar_evolution_enabled: bool = Field(True, description="Enable grammar evolution")
    semantic_pattern_discovery: bool = Field(True, description="Enable semantic pattern discovery")
    compositional_generation: bool = Field(True, description="Enable compositional generation")
    component_analysis_depth: Literal["basic", "deep"] = Field("deep", description="Analysis depth")
    linguistic_feature_extraction: bool = Field(True, description="Extract linguistic features")
    discriminative_pattern_threshold: float = Field(
        0.6, ge=0.3, le=0.9, description="Threshold for discriminative patterns"
    )
    
    # Advanced settings
    max_prompt_length: int = Field(2000, ge=100, le=10000, description="Maximum prompt length")
    min_component_frequency: int = Field(2, ge=1, le=10, description="Minimum component frequency")
    pattern_cache_size: int = Field(1000, ge=100, le=10000, description="Pattern cache size")
    evolution_temperature: float = Field(0.7, ge=0.1, le=2.0, description="Evolution randomness")


class CoordinationConfig(BaseModel):
    """Enhanced configuration for component coordination."""
    conflict_resolution_enabled: bool = Field(True, description="Enable conflict resolution")
    async_execution: bool = Field(True, description="Enable asynchronous execution")
    resource_allocation_strategy: Literal["fixed", "adaptive", "priority"] = Field(
        "adaptive", description="Resource allocation strategy"
    )
    priority_scheduling: bool = Field(True, description="Enable priority-based scheduling")
    complexity_management: bool = Field(True, description="Enable complexity management")
    regularization_enabled: bool = Field(True, description="Enable meta-learning regularization")
    
    # Advanced settings
    max_coordination_cycles: int = Field(100, ge=10, le=1000, description="Max coordination cycles")
    timeout_seconds: float = Field(30.0, ge=1.0, le=300.0, description="Coordination timeout")
    retry_attempts: int = Field(3, ge=1, le=10, description="Number of retry attempts")
    deadlock_detection: bool = Field(True, description="Enable deadlock detection")


class MetaOrchestratorConfig(BaseModel):
    """Enhanced MetaOrchestrator configuration with profile support."""
    
    # Component configurations
    rl_config: RLConfig = Field(default_factory=RLConfig)
    topology_config: TopologyConfig = Field(default_factory=TopologyConfig)
    hyperopt_config: HyperOptConfig = Field(default_factory=HyperOptConfig)
    prompt_config: PromptConfig = Field(default_factory=PromptConfig)
    coordination_config: CoordinationConfig = Field(default_factory=CoordinationConfig)
    
    # Global settings
    enabled: bool = Field(False, description="Enable MetaOrchestrator (backward compatibility)")
    optimization_mode: OptimizationMode = Field(OptimizationMode.BALANCED, description="Optimization mode")
    budget_allocation_strategy: BudgetStrategy = Field(BudgetStrategy.DYNAMIC, description="Budget strategy")
    performance_threshold: float = Field(0.05, ge=0.001, le=0.5, description="Min improvement threshold")
    max_optimization_rounds: int = Field(100, ge=10, le=1000, description="Maximum optimization rounds")
    
    # Resource management
    max_parallel_components: int = Field(4, ge=1, le=10, description="Max parallel components")
    total_compute_budget: float = Field(100.0, ge=10.0, le=1000.0, description="Total compute budget")
    resource_prediction_enabled: bool = Field(True, description="Enable resource prediction")
    approximation_fallback: bool = Field(True, description="Use approximations when needed")
    memory_limit_mb: Optional[int] = Field(None, ge=512, description="Memory limit in MB")
    
    # Performance and monitoring
    detailed_logging: bool = Field(True, description="Enable detailed logging")
    performance_tracking: bool = Field(True, description="Track performance metrics")
    component_metrics: bool = Field(True, description="Collect component metrics")
    export_metrics: bool = Field(False, description="Export metrics to external systems")
    metrics_export_interval: int = Field(60, ge=10, le=3600, description="Metrics export interval (seconds)")
    
    # Advanced features
    auto_tuning_enabled: bool = Field(False, description="Enable automatic configuration tuning")
    adaptive_timeout: bool = Field(True, description="Use adaptive timeouts")
    checkpoint_enabled: bool = Field(False, description="Enable optimization checkpointing")
    checkpoint_interval: int = Field(100, ge=10, le=1000, description="Checkpoint interval")
    
    # Environment integration
    config_file_path: Optional[str] = Field(None, description="Path to config file")
    environment_prefix: str = Field("GEPA_META", description="Environment variable prefix")
    profile: Optional[str] = Field(None, description="Configuration profile name")
    
    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
    
    @model_validator(mode='before')
    @classmethod
    def validate_config_consistency(cls, values):
        """Validate configuration consistency."""
        if isinstance(values, dict):
            optimization_mode = values.get('optimization_mode')
            budget_strategy = values.get('budget_allocation_strategy')
            
            # Warn about potentially conflicting settings
            if optimization_mode == OptimizationMode.CONSERVATIVE and budget_strategy == BudgetStrategy.ADAPTIVE:
                # This is fine, just might be suboptimal
                pass
                
            if optimization_mode == OptimizationMode.AGGRESSIVE and not values.get('approximation_fallback'):
                raise ValueError("Aggressive mode requires approximation fallback for resource management")
        
        return values
    
    @classmethod
    def from_profile(cls, profile_name: str) -> 'MetaOrchestratorConfig':
        """Create configuration from predefined profile."""
        return ConfigProfiles.get_profile(profile_name)
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'MetaOrchestratorConfig':
        """Load configuration from file (JSON or YAML)."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls.parse_obj(data)
    
    @classmethod
    def from_env(cls, prefix: str = "GEPA_META") -> 'MetaOrchestratorConfig':
        """Load configuration from environment variables."""
        config_data = {}
        
        # Map environment variables to config fields
        env_mapping = {
            f"{prefix}_ENABLED": "enabled",
            f"{prefix}_MODE": "optimization_mode", 
            f"{prefix}_BUDGET_STRATEGY": "budget_allocation_strategy",
            f"{prefix}_PERFORMANCE_THRESHOLD": "performance_threshold",
            f"{prefix}_MAX_ROUNDS": "max_optimization_rounds",
            f"{prefix}_COMPUTE_BUDGET": "total_compute_budget",
            f"{prefix}_MEMORY_LIMIT": "memory_limit_mb",
            f"{prefix}_DETAILED_LOGGING": "detailed_logging",
            f"{prefix}_PROFILE": "profile"
        }
        
        for env_var, config_key in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                # Type conversion
                if config_key in ["enabled", "detailed_logging", "performance_tracking"]:
                    config_data[config_key] = value.lower() in ['true', '1', 'yes', 'on']
                elif config_key in ["performance_threshold", "total_compute_budget"]:
                    config_data[config_key] = float(value)
                elif config_key in ["max_optimization_rounds", "memory_limit_mb"]:
                    config_data[config_key] = int(value)
                else:
                    config_data[config_key] = value
        
        return cls.parse_obj(config_data)
    
    def to_file(self, file_path: Union[str, Path], format: str = "yaml") -> None:
        """Save configuration to file."""
        file_path = Path(file_path)
        
        data = self.dict()
        
        with open(file_path, 'w') as f:
            if format.lower() == "yaml":
                yaml.dump(data, f, default_flow_style=False, indent=2)
            else:
                json.dump(data, f, indent=2)
    
    def update_from_dict(self, updates: Dict[str, Any]) -> 'MetaOrchestratorConfig':
        """Create updated configuration from dictionary."""
        current_dict = self.dict()
        current_dict.update(updates)
        return self.__class__.parse_obj(current_dict)
    
    def get_resource_allocation(self) -> Dict[str, float]:
        """Get resource allocation percentages based on optimization mode."""
        if self.optimization_mode == OptimizationMode.EXPLORATION:
            return {"rl_selector": 0.3, "topology_evolver": 0.4, "hyperopt": 0.2, "prompt_evolver": 0.1}
        elif self.optimization_mode == OptimizationMode.EXPLOITATION:
            return {"rl_selector": 0.4, "topology_evolver": 0.1, "hyperopt": 0.4, "prompt_evolver": 0.1}
        elif self.optimization_mode == OptimizationMode.CONSERVATIVE:
            return {"rl_selector": 0.35, "topology_evolver": 0.15, "hyperopt": 0.35, "prompt_evolver": 0.15}
        elif self.optimization_mode == OptimizationMode.AGGRESSIVE:
            return {"rl_selector": 0.25, "topology_evolver": 0.35, "hyperopt": 0.25, "prompt_evolver": 0.15}
        else:  # BALANCED
            return {"rl_selector": 0.3, "topology_evolver": 0.25, "hyperopt": 0.3, "prompt_evolver": 0.15}


class ConfigProfiles:
    """Predefined configuration profiles for different use cases."""
    
    @staticmethod
    def get_profile(profile_name: str) -> MetaOrchestratorConfig:
        """Get configuration for a specific profile."""
        profiles = {
            "development": ConfigProfiles._development_profile(),
            "production": ConfigProfiles._production_profile(),
            "research": ConfigProfiles._research_profile(),
            "conservative": ConfigProfiles._conservative_profile(),
            "aggressive": ConfigProfiles._aggressive_profile(),
            "minimal": ConfigProfiles._minimal_profile()
        }
        
        if profile_name not in profiles:
            available = ", ".join(profiles.keys())
            raise ValueError(f"Unknown profile '{profile_name}'. Available: {available}")
        
        return profiles[profile_name]
    
    @staticmethod
    def _development_profile() -> MetaOrchestratorConfig:
        """Development profile with moderate settings."""
        return MetaOrchestratorConfig(
            enabled=True,
            optimization_mode=OptimizationMode.BALANCED,
            budget_allocation_strategy=BudgetStrategy.DYNAMIC,
            max_optimization_rounds=50,
            total_compute_budget=50.0,
            detailed_logging=True,
            checkpoint_enabled=True,
            checkpoint_interval=25
        )
    
    @staticmethod 
    def _production_profile() -> MetaOrchestratorConfig:
        """Production profile optimized for performance."""
        return MetaOrchestratorConfig(
            enabled=True,
            optimization_mode=OptimizationMode.EXPLOITATION,
            budget_allocation_strategy=BudgetStrategy.ADAPTIVE,
            max_optimization_rounds=200,
            total_compute_budget=200.0,
            detailed_logging=False,
            performance_tracking=True,
            approximation_fallback=True,
            adaptive_timeout=True
        )
    
    @staticmethod
    def _research_profile() -> MetaOrchestratorConfig:
        """Research profile with extensive exploration."""
        return MetaOrchestratorConfig(
            enabled=True,
            optimization_mode=OptimizationMode.EXPLORATION,
            budget_allocation_strategy=BudgetStrategy.ADAPTIVE,
            max_optimization_rounds=500,
            total_compute_budget=500.0,
            detailed_logging=True,
            component_metrics=True,
            export_metrics=True,
            checkpoint_enabled=True,
            checkpoint_interval=50,
            topology_config=TopologyConfig(
                diversity_maintenance=True,
                mutation_rate=0.4,
                crossover_rate=0.6
            )
        )
    
    @staticmethod
    def _conservative_profile() -> MetaOrchestratorConfig:
        """Conservative profile with minimal resource usage."""
        return MetaOrchestratorConfig(
            enabled=True,
            optimization_mode=OptimizationMode.CONSERVATIVE,
            budget_allocation_strategy=BudgetStrategy.FIXED,
            max_optimization_rounds=30,
            total_compute_budget=30.0,
            max_parallel_components=2,
            approximation_fallback=True,
            detailed_logging=False
        )
    
    @staticmethod
    def _aggressive_profile() -> MetaOrchestratorConfig:
        """Aggressive profile with maximum resource usage."""
        return MetaOrchestratorConfig(
            enabled=True,
            optimization_mode=OptimizationMode.AGGRESSIVE,
            budget_allocation_strategy=BudgetStrategy.ADAPTIVE,
            max_optimization_rounds=1000,
            total_compute_budget=1000.0,
            max_parallel_components=8,
            approximation_fallback=True,
            auto_tuning_enabled=True,
            topology_config=TopologyConfig(
                mutation_rate=0.5,
                crossover_rate=0.8,
                elitism_rate=0.05
            )
        )
    
    @staticmethod
    def _minimal_profile() -> MetaOrchestratorConfig:
        """Minimal profile for testing and quick experiments."""
        return MetaOrchestratorConfig(
            enabled=True,
            optimization_mode=OptimizationMode.BALANCED,
            budget_allocation_strategy=BudgetStrategy.FIXED,
            max_optimization_rounds=10,
            total_compute_budget=10.0,
            max_parallel_components=1,
            detailed_logging=False,
            performance_tracking=False,
            component_metrics=False
        )