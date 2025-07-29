"""Configuration management for GEPA."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator
import yaml


class InferenceConfig(BaseModel):
    """Configuration for inference providers."""
    
    provider: str = Field(..., description="Provider name (openai, anthropic, etc.)")
    model: str = Field(..., description="Model identifier")
    api_key: Optional[str] = Field(None, description="API key for provider")
    base_url: Optional[str] = Field(None, description="Custom base URL for local providers")
    max_tokens: int = Field(4096, description="Maximum tokens in response")
    temperature: float = Field(0.7, description="Sampling temperature")
    timeout: float = Field(30.0, description="Request timeout in seconds")
    retry_attempts: int = Field(3, description="Number of retry attempts")
    
    @validator('temperature')
    def temperature_must_be_valid(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v


class DatabaseConfig(BaseModel):
    """Configuration for database connection."""
    
    url: str = Field("postgresql://gepa:gepa@localhost/gepa", description="Database URL")
    pool_size: int = Field(10, description="Connection pool size")
    max_overflow: int = Field(20, description="Maximum overflow connections")
    echo: bool = Field(False, description="Echo SQL statements")


class OptimizationConfig(BaseModel):
    """Configuration for GEPA optimization."""
    
    budget: int = Field(100, description="Maximum number of rollouts")
    minibatch_size: int = Field(5, description="Size of evaluation minibatches")
    pareto_set_size: int = Field(10, description="Size of Pareto frontier")
    reflection_model: Optional[str] = Field(None, description="Model for reflection (uses main if None)")
    enable_crossover: bool = Field(True, description="Enable crossover operations")
    crossover_probability: float = Field(0.3, description="Probability of crossover")
    mutation_types: List[str] = Field(
        ["rewrite", "insert", "delete", "compress"],
        description="Types of mutations to use"
    )
    
    @validator('crossover_probability')
    def crossover_probability_must_be_valid(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Crossover probability must be between 0.0 and 1.0')
        return v


class ObservabilityConfig(BaseModel):
    """Configuration for observability and metrics."""
    
    enable_prometheus: bool = Field(True, description="Enable Prometheus metrics")
    prometheus_port: int = Field(8000, description="Prometheus metrics port")
    enable_tracing: bool = Field(True, description="Enable OpenTelemetry tracing")
    log_level: str = Field("INFO", description="Logging level")
    log_format: str = Field("json", description="Log format (json or console)")


class GEPAConfig(BaseModel):
    """Main GEPA configuration."""
    
    inference: InferenceConfig
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    
    @classmethod
    def from_file(cls, path: Path) -> "GEPAConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GEPAConfig":
        """Load configuration from dictionary."""
        return cls(**data)
    
    def to_file(self, path: Path) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False)