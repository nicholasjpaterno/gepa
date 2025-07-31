"""
MetaOrchestrator Configuration Examples

This module demonstrates various ways to configure the MetaOrchestrator
for different use cases, environments, and optimization goals.
"""

import os
import json
import tempfile
from pathlib import Path

from src.gepa.meta_orchestrator import MetaOrchestratorConfig, ConfigProfiles
from src.gepa.meta_orchestrator.config import (
    OptimizationMode, 
    BudgetStrategy,
    RLConfig,
    TopologyConfig,
    HyperOptConfig,
    PromptConfig,
    CoordinationConfig
)


def example_1_basic_configuration():
    """Example 1: Basic configuration with default settings."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Configuration")
    print("=" * 60)
    
    # Create basic configuration
    config = MetaOrchestratorConfig(
        enabled=True,
        optimization_mode=OptimizationMode.BALANCED,
        max_optimization_rounds=50,
        total_compute_budget=100.0
    )
    
    print(f"Optimization Mode: {config.optimization_mode}")
    print(f"Budget Strategy: {config.budget_allocation_strategy}")
    print(f"Max Rounds: {config.max_optimization_rounds}")
    print(f"Compute Budget: {config.total_compute_budget}")
    print(f"Detailed Logging: {config.detailed_logging}")
    
    # Show resource allocation
    allocation = config.get_resource_allocation()
    print("\nResource Allocation:")
    for component, percentage in allocation.items():
        print(f"  {component}: {percentage:.1%}")
    
    return config


def example_2_predefined_profiles():
    """Example 2: Using predefined configuration profiles."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Predefined Configuration Profiles")
    print("=" * 60)
    
    profiles = ["development", "production", "research", "conservative", "aggressive", "minimal"]
    
    for profile_name in profiles:
        print(f"\n--- {profile_name.upper()} Profile ---")
        config = ConfigProfiles.get_profile(profile_name)
        
        print(f"Optimization Mode: {config.optimization_mode}")
        print(f"Budget Strategy: {config.budget_allocation_strategy}")
        print(f"Max Rounds: {config.max_optimization_rounds}")
        print(f"Compute Budget: {config.total_compute_budget}")
        print(f"Parallel Components: {config.max_parallel_components}")
        print(f"Detailed Logging: {config.detailed_logging}")
        
        # Show what makes this profile unique
        if profile_name == "development":
            print("✓ Moderate settings with checkpointing enabled")
        elif profile_name == "production":
            print("✓ Optimized for performance with minimal logging")
        elif profile_name == "research":
            print("✓ Extensive exploration with metrics export")
        elif profile_name == "conservative":
            print("✓ Minimal resource usage for constrained environments")
        elif profile_name == "aggressive":
            print("✓ Maximum resource usage with auto-tuning")
        elif profile_name == "minimal":
            print("✓ Quick experiments with minimal overhead")
    
    return ConfigProfiles.get_profile("development")


def example_3_custom_component_configuration():
    """Example 3: Custom component-specific configurations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom Component Configuration")
    print("=" * 60)
    
    # Custom RL configuration for high-performance scenarios
    custom_rl_config = RLConfig(
        state_dim=15,                    # Extended state representation
        action_dim=8,                    # More algorithm choices
        hidden_dims=[512, 512, 256],     # Larger networks
        learning_rate=1e-4,              # Lower learning rate for stability
        buffer_capacity=50000,           # Larger experience buffer
        batch_size=128,                  # Larger batches
        gamma=0.995,                     # Longer-term planning
        exploration_noise=0.05,          # Lower exploration for exploitation
        prioritized_replay=True,         # Use prioritized experience replay
        double_q_learning=True,          # Enable double Q-learning
        dueling_network=True             # Use dueling architecture
    )
    
    # Custom topology configuration for aggressive evolution
    custom_topology_config = TopologyConfig(
        max_complexity_threshold=5.0,    # Allow more complex systems
        min_topology_budget=15,          # Higher minimum budget
        min_improvement_threshold=0.02,  # Lower threshold for evolution
        mutation_rate=0.4,               # Higher mutation rate
        crossover_rate=0.8,              # Higher crossover rate
        elitism_rate=0.05,               # Lower elitism (more exploration)
        performance_prediction_enabled=True,
        diversity_maintenance=True
    )
    
    # Custom hyperparameter optimization configuration
    custom_hyperopt_config = HyperOptConfig(
        acquisition_function="upper_confidence_bound",  # Different acquisition function
        n_initial_points=10,             # More initial exploration
        transfer_learning_enabled=True,  # Enable transfer learning
        multi_fidelity_enabled=True,     # Use multi-fidelity optimization
        max_similar_contexts=20,         # More context for transfer
        similarity_threshold=0.6,        # Lower threshold for similarity
        gp_kernel="matern32",            # Different kernel
        exploration_weight=0.2,          # Higher exploration weight
        optimize_hyperparameters=True    # Optimize GP hyperparameters
    )
    
    # Custom prompt configuration for advanced evolution
    custom_prompt_config = PromptConfig(
        grammar_evolution_enabled=True,
        semantic_pattern_discovery=True,
        compositional_generation=True,
        component_analysis_depth="deep",
        discriminative_pattern_threshold=0.5,  # Lower threshold
        max_prompt_length=3000,                 # Longer prompts allowed
        evolution_temperature=0.8               # Higher temperature for more variation
    )
    
    # Custom coordination configuration
    custom_coordination_config = CoordinationConfig(
        conflict_resolution_enabled=True,
        async_execution=True,
        resource_allocation_strategy="adaptive",
        priority_scheduling=True,
        complexity_management=True,
        regularization_enabled=True,
        max_coordination_cycles=200,     # More coordination cycles
        timeout_seconds=60.0,            # Longer timeout
        retry_attempts=5,                # More retry attempts
        deadlock_detection=True
    )
    
    # Create configuration with custom components
    config = MetaOrchestratorConfig(
        enabled=True,
        optimization_mode=OptimizationMode.AGGRESSIVE,
        budget_allocation_strategy=BudgetStrategy.ADAPTIVE,
        max_optimization_rounds=200,
        total_compute_budget=500.0,
        
        # Custom component configurations
        rl_config=custom_rl_config,
        topology_config=custom_topology_config,
        hyperopt_config=custom_hyperopt_config,
        prompt_config=custom_prompt_config,
        coordination_config=custom_coordination_config,
        
        # Advanced features
        auto_tuning_enabled=True,
        checkpoint_enabled=True,
        adaptive_timeout=True,
        export_metrics=True,
        
        # Resource management
        max_parallel_components=6,
        memory_limit_mb=2048,
        approximation_fallback=True
    )
    
    print("Custom Configuration Created:")
    print(f"  RL Network: {config.rl_config.hidden_dims}")
    print(f"  RL Buffer: {config.rl_config.buffer_capacity:,}")
    print(f"  Topology Mutations: {config.topology_config.mutation_rate:.1%}")
    print(f"  HyperOpt Acquisition: {config.hyperopt_config.acquisition_function}")
    print(f"  Prompt Max Length: {config.prompt_config.max_prompt_length:,}")
    print(f"  Coordination Timeout: {config.coordination_config.timeout_seconds}s")
    print(f"  Memory Limit: {config.memory_limit_mb}MB")
    
    return config


def example_4_environment_based_configuration():
    """Example 4: Environment-based configuration loading."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Environment-Based Configuration")
    print("=" * 60)
    
    # Set environment variables for demonstration
    env_vars = {
        'GEPA_META_ENABLED': 'true',
        'GEPA_META_MODE': 'exploration',
        'GEPA_META_BUDGET_STRATEGY': 'adaptive',
        'GEPA_META_PERFORMANCE_THRESHOLD': '0.03',
        'GEPA_META_MAX_ROUNDS': '150',
        'GEPA_META_COMPUTE_BUDGET': '300.0',
        'GEPA_META_MEMORY_LIMIT': '1024',
        'GEPA_META_DETAILED_LOGGING': 'true'
    }
    
    # Temporarily set environment variables
    original_env = {}
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        # Load configuration from environment
        config = MetaOrchestratorConfig.from_env("GEPA_META")
        
        print("Configuration loaded from environment variables:")
        print(f"  Enabled: {config.enabled}")
        print(f"  Mode: {config.optimization_mode}")
        print(f"  Budget Strategy: {config.budget_allocation_strategy}")
        print(f"  Performance Threshold: {config.performance_threshold}")
        print(f"  Max Rounds: {config.max_optimization_rounds}")
        print(f"  Compute Budget: {config.total_compute_budget}")
        print(f"  Memory Limit: {config.memory_limit_mb}MB")
        print(f"  Detailed Logging: {config.detailed_logging}")
        
        print("\nEnvironment Variables Used:")
        for key in env_vars:
            print(f"  {key}={os.environ[key]}")
    
    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is not None:
                os.environ[key] = original_value
            elif key in os.environ:
                del os.environ[key]
    
    return config


def example_5_file_based_configuration():
    """Example 5: File-based configuration (JSON and YAML)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: File-Based Configuration")
    print("=" * 60)
    
    # Example JSON configuration
    json_config = {
        "enabled": True,
        "optimization_mode": "balanced",
        "budget_allocation_strategy": "dynamic",
        "max_optimization_rounds": 100,
        "total_compute_budget": 200.0,
        "detailed_logging": True,
        "checkpoint_enabled": True,
        "auto_tuning_enabled": False,
        "rl_config": {
            "learning_rate": 0.0005,
            "buffer_capacity": 20000,
            "batch_size": 64,
            "prioritized_replay": True
        },
        "topology_config": {
            "mutation_rate": 0.35,
            "crossover_rate": 0.75,
            "max_complexity_threshold": 3.0
        },
        "hyperopt_config": {
            "acquisition_function": "expected_improvement",
            "n_initial_points": 8,
            "transfer_learning_enabled": True
        },
        "prompt_config": {
            "grammar_evolution_enabled": True,
            "semantic_pattern_discovery": True,
            "discriminative_pattern_threshold": 0.65
        },
        "coordination_config": {
            "async_execution": True,
            "resource_allocation_strategy": "adaptive",
            "timeout_seconds": 45.0
        }
    }
    
    # Save to temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(json_config, f, indent=2)
        json_file = f.name
    
    try:
        # Load from JSON file
        config_from_json = MetaOrchestratorConfig.from_file(json_file)
        
        print("Configuration loaded from JSON file:")
        print(f"  File: {json_file}")
        print(f"  Mode: {config_from_json.optimization_mode}")
        print(f"  Budget: {config_from_json.total_compute_budget}")
        print(f"  RL Learning Rate: {config_from_json.rl_config.learning_rate}")
        print(f"  Topology Mutation Rate: {config_from_json.topology_config.mutation_rate}")
        print(f"  HyperOpt Initial Points: {config_from_json.hyperopt_config.n_initial_points}")
        
        # Save configuration back to file (demonstrate round-trip)
        output_file = json_file.replace('.json', '_output.yaml')
        config_from_json.to_file(output_file, format='yaml')
        
        print(f"\nConfiguration saved to: {output_file}")
        
        # Show file content (first few lines)
        with open(output_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')[:10]
            print("YAML Output (first 10 lines):")
            for line in lines:
                print(f"  {line}")
        
        return config_from_json
    
    finally:
        # Clean up temporary files
        Path(json_file).unlink(missing_ok=True)
        Path(json_file.replace('.json', '_output.yaml')).unlink(missing_ok=True)


def example_6_configuration_validation_and_updates():
    """Example 6: Configuration validation and dynamic updates."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Configuration Validation and Updates")
    print("=" * 60)
    
    # Start with a base configuration
    base_config = ConfigProfiles.get_profile("development")
    
    print("Base Configuration (Development Profile):")
    print(f"  Mode: {base_config.optimization_mode}")
    print(f"  Rounds: {base_config.max_optimization_rounds}")
    print(f"  Budget: {base_config.total_compute_budget}")
    
    # Example of valid updates
    print("\n--- Valid Configuration Updates ---")
    
    valid_updates = {
        "max_optimization_rounds": 75,
        "total_compute_budget": 150.0,
        "detailed_logging": False,
        "checkpoint_interval": 20
    }
    
    updated_config = base_config.update_from_dict(valid_updates)
    
    print("Updated Configuration:")
    print(f"  Rounds: {updated_config.max_optimization_rounds}")
    print(f"  Budget: {updated_config.total_compute_budget}")
    print(f"  Logging: {updated_config.detailed_logging}")
    print(f"  Checkpoint Interval: {updated_config.checkpoint_interval}")
    
    # Example of configuration validation errors
    print("\n--- Configuration Validation Examples ---")
    
    try:
        # This should fail - aggressive mode without approximation fallback
        invalid_config = MetaOrchestratorConfig(
            optimization_mode=OptimizationMode.AGGRESSIVE,
            approximation_fallback=False
        )
    except ValueError as e:
        print(f"✓ Validation Error Caught: {e}")
    
    try:
        # This should fail - invalid performance threshold
        invalid_config = MetaOrchestratorConfig(
            performance_threshold=2.0  # Should be <= 0.5
        )
    except ValueError as e:
        print(f"✓ Validation Error Caught: {e}")
    
    try:
        # This should fail - invalid RL configuration
        invalid_rl_config = RLConfig(
            hidden_dims=[32000]  # Too large, should be <= 1024
        )
    except ValueError as e:
        print(f"✓ Validation Error Caught: {e}")
    
    # Example of component-specific validation
    print("\n--- Component Validation Examples ---")
    
    # Valid topology configuration
    valid_topology = TopologyConfig(
        mutation_rate=0.4,
        crossover_rate=0.7,
        mutation_types=["add_module", "remove_module"]
    )
    print(f"✓ Valid Topology Config: mutation_rate={valid_topology.mutation_rate}")
    
    try:
        # Invalid topology configuration
        invalid_topology = TopologyConfig(
            mutation_types=["invalid_mutation_type"]
        )
    except ValueError as e:
        print(f"✓ Topology Validation Error Caught: {e}")
    
    return updated_config


def example_7_optimization_mode_comparison():
    """Example 7: Comparison of different optimization modes."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Optimization Mode Comparison")
    print("=" * 60)
    
    modes = [
        OptimizationMode.EXPLORATION,
        OptimizationMode.EXPLOITATION,
        OptimizationMode.BALANCED,
        OptimizationMode.CONSERVATIVE,
        OptimizationMode.AGGRESSIVE
    ]
    
    for mode in modes:
        config = MetaOrchestratorConfig(
            enabled=True,
            optimization_mode=mode,
            total_compute_budget=100.0
        )
        
        allocation = config.get_resource_allocation()
        
        print(f"\n--- {mode.upper()} Mode ---")
        print(f"Description: {_get_mode_description(mode)}")
        print("Resource Allocation:")
        for component, percentage in allocation.items():
            print(f"  {component}: {percentage:.1%}")
        
        print("Best For:")
        best_for = _get_mode_best_for(mode)
        for use_case in best_for:
            print(f"  • {use_case}")


def _get_mode_description(mode: OptimizationMode) -> str:
    """Get description for optimization mode."""
    descriptions = {
        OptimizationMode.EXPLORATION: "Focus on discovering new solutions and system architectures",
        OptimizationMode.EXPLOITATION: "Focus on improving current best solutions with fine-tuning",
        OptimizationMode.BALANCED: "Balance between exploration of new solutions and exploitation of current best",
        OptimizationMode.CONSERVATIVE: "Minimal resource usage with safe, incremental improvements",
        OptimizationMode.AGGRESSIVE: "Maximum resource usage for breakthrough performance improvements"
    }
    return descriptions.get(mode, "Unknown mode")


def _get_mode_best_for(mode: OptimizationMode) -> list:
    """Get use cases each mode is best for."""
    use_cases = {
        OptimizationMode.EXPLORATION: [
            "Research and development",
            "Novel problem domains",
            "Finding innovative solutions",
            "Early-stage optimization"
        ],
        OptimizationMode.EXPLOITATION: [
            "Production optimization",
            "Fine-tuning existing systems",
            "Performance maximization",
            "Late-stage optimization"
        ],
        OptimizationMode.BALANCED: [
            "General-purpose optimization",
            "Mixed exploration and refinement",
            "Unknown problem characteristics",
            "Development environments"
        ],
        OptimizationMode.CONSERVATIVE: [
            "Resource-constrained environments",
            "Risk-averse scenarios",
            "Stable, incremental improvements",
            "Production with strict limits"
        ],
        OptimizationMode.AGGRESSIVE: [
            "High-performance computing environments",
            "Breakthrough performance needed",
            "Abundant computational resources",
            "Competitive optimization scenarios"
        ]
    }
    return use_cases.get(mode, [])


def main():
    """Run all configuration examples."""
    print("MetaOrchestrator Configuration Examples")
    print("=" * 80)
    
    # Run all examples
    example_1_basic_configuration()
    example_2_predefined_profiles()
    example_3_custom_component_configuration()
    example_4_environment_based_configuration()
    example_5_file_based_configuration()
    example_6_configuration_validation_and_updates()
    example_7_optimization_mode_comparison()
    
    print("\n" + "=" * 80)
    print("All Configuration Examples Complete!")
    print("=" * 80)
    
    print("\nKey Takeaways:")
    print("• Use predefined profiles for common scenarios")
    print("• Customize component configs for specific needs")
    print("• Load from environment variables for deployment")
    print("• Save/load configurations from files for reproducibility")
    print("• Configuration validation prevents common errors")
    print("• Different optimization modes suit different use cases")
    
    print("\nNext Steps:")
    print("• Choose the appropriate profile for your use case")
    print("• Customize component settings based on your requirements")
    print("• Set up environment variables for production deployment")
    print("• Create configuration files for different environments")
    print("• Test your configuration with the MetaOrchestrator demo")


if __name__ == "__main__":
    main()