"""GEPA configuration management."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import os


@dataclass
class InferenceConfig:
    """Configuration for inference clients."""
    provider: str = "openai"  # openai, anthropic, ollama, lmstudio
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: int = 30
    retry_attempts: int = 3


@dataclass
class OptimizationConfig:
    """Configuration for optimization process."""
    budget: int = 100
    minibatch_size: int = 5
    pareto_set_size: int = 10
    max_generations: Optional[int] = None
    reflection_model: Optional[str] = None
    enable_system_aware_merge: bool = True
    merge_probability: float = 0.3
    enable_crossover: bool = True
    crossover_probability: float = 0.3
    mutation_types: List[str] = None
    
    def __post_init__(self):
        if self.mutation_types is None:
            self.mutation_types = ["rewrite", "insert", "delete", "compress"]


@dataclass
class DatabaseConfig:
    """Configuration for database connection."""
    url: str = "sqlite:///gepa.db"
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class ObservabilityConfig:
    """Configuration for observability features."""
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_metrics: bool = True
    enable_tracing: bool = False
    metrics_port: int = 8000


@dataclass
class AdvancedAlgorithmConfig:
    """Configuration for advanced algorithm implementations."""
    
    # Algorithm 2 improvements (Pareto candidate sampling)
    enable_score_prediction: bool = True
    score_prediction_method: str = "ensemble"  # similarity, pattern, ensemble
    enable_adaptive_comparison: bool = True
    comparison_confidence_threshold: float = 0.95
    
    # Algorithm 3 improvements (Reflective prompt mutation)
    module_selection_strategy: str = "intelligent"  # round_robin, intelligent
    enable_bandit_selection: bool = True
    bandit_exploration_factor: float = 1.4
    minibatch_strategy: str = "strategic"  # random, strategic
    enable_difficulty_sampling: bool = True
    enable_diversity_sampling: bool = True
    
    # Algorithm 4 improvements (System aware merge)
    compatibility_analysis_depth: str = "deep"  # basic, deep
    enable_semantic_similarity: bool = True
    enable_style_analysis: bool = True
    enable_statistical_testing: bool = True
    enable_risk_assessment: bool = True
    enable_mcda_scoring: bool = True
    
    # Learning and adaptation
    enable_historical_learning: bool = True
    adaptation_rate: float = 0.1
    learning_window_size: int = 20
    
    # Performance and debugging
    enable_caching: bool = True
    cache_size: int = 1000
    enable_performance_monitoring: bool = False
    debug_mode: bool = False


@dataclass  
class GEPAConfig:
    """Main GEPA configuration."""
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    advanced: AdvancedAlgorithmConfig = field(default_factory=AdvancedAlgorithmConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GEPAConfig":
        """Create config from dictionary."""
        return cls(
            inference=InferenceConfig(**config_dict.get("inference", {})),
            optimization=OptimizationConfig(**config_dict.get("optimization", {})),
            database=DatabaseConfig(**config_dict.get("database", {})),
            observability=ObservabilityConfig(**config_dict.get("observability", {})),
            advanced=AdvancedAlgorithmConfig(**config_dict.get("advanced", {}))
        )
    
    @classmethod
    def from_env(cls) -> "GEPAConfig":
        """Create config from environment variables."""
        return cls(
            inference=InferenceConfig(
                provider=os.getenv("GEPA_PROVIDER", "openai"),
                model=os.getenv("GEPA_MODEL", "gpt-3.5-turbo"),
                api_key=os.getenv("GEPA_API_KEY"),
                base_url=os.getenv("GEPA_BASE_URL"),
                temperature=float(os.getenv("GEPA_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("GEPA_MAX_TOKENS", "0")) or None,
                timeout=int(os.getenv("GEPA_TIMEOUT", "30"))
            ),
            optimization=OptimizationConfig(
                budget=int(os.getenv("GEPA_BUDGET", "100")),
                minibatch_size=int(os.getenv("GEPA_MINIBATCH_SIZE", "5")),
                pareto_set_size=int(os.getenv("GEPA_PARETO_SET_SIZE", "10")),
                max_generations=int(os.getenv("GEPA_MAX_GENERATIONS", "0")) or None,
                reflection_model=os.getenv("GEPA_REFLECTION_MODEL"),
                enable_system_aware_merge=os.getenv("GEPA_ENABLE_MERGE", "true").lower() == "true",
                merge_probability=float(os.getenv("GEPA_MERGE_PROBABILITY", "0.3"))
            ),
            database=DatabaseConfig(
                url=os.getenv("GEPA_DATABASE_URL", "sqlite:///gepa.db"),
                echo=os.getenv("GEPA_DATABASE_ECHO", "false").lower() == "true",
                pool_size=int(os.getenv("GEPA_DATABASE_POOL_SIZE", "10")),
                max_overflow=int(os.getenv("GEPA_DATABASE_MAX_OVERFLOW", "20"))
            ),
            observability=ObservabilityConfig(
                enable_logging=os.getenv("GEPA_ENABLE_LOGGING", "true").lower() == "true",
                log_level=os.getenv("GEPA_LOG_LEVEL", "INFO"),
                enable_metrics=os.getenv("GEPA_ENABLE_METRICS", "true").lower() == "true",
                enable_tracing=os.getenv("GEPA_ENABLE_TRACING", "false").lower() == "true",
                metrics_port=int(os.getenv("GEPA_METRICS_PORT", "8000"))
            ),
            advanced=AdvancedAlgorithmConfig(
                enable_score_prediction=os.getenv("GEPA_ENABLE_SCORE_PREDICTION", "true").lower() == "true",
                score_prediction_method=os.getenv("GEPA_SCORE_PREDICTION_METHOD", "ensemble"),
                enable_adaptive_comparison=os.getenv("GEPA_ENABLE_ADAPTIVE_COMPARISON", "true").lower() == "true",
                module_selection_strategy=os.getenv("GEPA_MODULE_SELECTION_STRATEGY", "intelligent"),
                minibatch_strategy=os.getenv("GEPA_MINIBATCH_STRATEGY", "strategic"),
                compatibility_analysis_depth=os.getenv("GEPA_COMPATIBILITY_ANALYSIS_DEPTH", "deep"),
                enable_statistical_testing=os.getenv("GEPA_ENABLE_STATISTICAL_TESTING", "true").lower() == "true",
                enable_risk_assessment=os.getenv("GEPA_ENABLE_RISK_ASSESSMENT", "true").lower() == "true",
                enable_historical_learning=os.getenv("GEPA_ENABLE_HISTORICAL_LEARNING", "true").lower() == "true"
            )
        )