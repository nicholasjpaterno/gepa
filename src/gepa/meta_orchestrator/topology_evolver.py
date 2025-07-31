"""NEAT-inspired Topology Evolution for MetaOrchestrator."""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..core.system import CompoundAISystem, LanguageModule, SequentialFlow, IOSchema
from .config import TopologyConfig
from .state import OptimizationState, PerformanceMetrics

logger = logging.getLogger(__name__)


class MutationType(Enum):
    """Types of topology mutations."""
    ADD_MODULE = "add_module"
    REMOVE_MODULE = "remove_module"
    RECONNECT = "reconnect"
    CONTROL_FLOW = "control_flow"


@dataclass
class TopologyMutation:
    """Represents a topology mutation."""
    mutation_type: MutationType
    target_module: Optional[str] = None
    new_module: Optional[LanguageModule] = None
    connection_changes: Optional[Dict[str, Any]] = None
    complexity_delta: float = 0.0
    predicted_improvement: float = 0.0


class ComplexityRegulator:
    """Manages system complexity to prevent bloat."""
    
    def __init__(self, max_complexity: float = 2.0):
        self.max_complexity = max_complexity
        self.complexity_history = []
    
    def analyze_efficiency(self, system: CompoundAISystem) -> float:
        """Analyze complexity-to-performance ratio."""
        complexity = self.calculate_complexity(system)
        
        # Simple complexity metric based on number of modules and connections
        base_complexity = len(system.modules)
        
        # Add complexity for sophisticated control flows
        if hasattr(system.control_flow, 'module_order'):
            control_complexity = len(system.control_flow.module_order) * 0.1
        else:
            control_complexity = 0.5  # Default for complex control flows
        
        total_complexity = base_complexity + control_complexity
        return total_complexity / self.max_complexity
    
    def calculate_complexity(self, system: CompoundAISystem) -> float:
        """Calculate system complexity score."""
        # Base complexity from number of modules
        module_complexity = len(system.modules)
        
        # Prompt complexity (average length)
        prompt_complexity = 0.0
        for module in system.modules.values():
            prompt_complexity += len(module.prompt.split()) / 100.0
        
        # Control flow complexity
        control_complexity = 1.0  # Base complexity
        if hasattr(system.control_flow, 'module_order'):
            control_complexity = len(system.control_flow.module_order) * 0.2
        
        return module_complexity + prompt_complexity + control_complexity
    
    def is_within_limits(self, complexity: float) -> bool:
        """Check if complexity is within acceptable limits."""
        return complexity <= self.max_complexity


class ArchitecturePerformancePredictor:
    """Predicts performance impact of architecture changes."""
    
    def __init__(self):
        self.performance_history = []
        self.mutation_outcomes = {}
    
    def predict_gain(
        self,
        system: CompoundAISystem,
        proposed_changes: List[TopologyMutation]
    ) -> float:
        """Predict performance gain from proposed changes."""
        if not proposed_changes:
            return 0.0
        
        total_predicted_gain = 0.0
        
        for mutation in proposed_changes:
            # Base prediction on mutation type and historical data
            mutation_key = f"{mutation.mutation_type.value}_{len(system.modules)}"
            
            if mutation_key in self.mutation_outcomes:
                outcomes = self.mutation_outcomes[mutation_key]
                predicted_gain = np.mean([outcome["improvement"] for outcome in outcomes])
            else:
                # Default predictions based on mutation type
                predicted_gain = self._get_default_prediction(mutation, system)
            
            total_predicted_gain += predicted_gain
        
        return total_predicted_gain
    
    def _get_default_prediction(
        self,
        mutation: TopologyMutation,
        system: CompoundAISystem
    ) -> float:
        """Get default prediction for new mutation types."""
        if mutation.mutation_type == MutationType.ADD_MODULE:
            # Adding modules can help but increases complexity
            return 0.05 if len(system.modules) < 5 else -0.02
        elif mutation.mutation_type == MutationType.REMOVE_MODULE:
            # Removing modules reduces complexity but may hurt performance
            return -0.03 if len(system.modules) <= 2 else 0.02
        elif mutation.mutation_type == MutationType.RECONNECT:
            # Reconnection has variable impact
            return random.uniform(-0.02, 0.04)
        elif mutation.mutation_type == MutationType.CONTROL_FLOW:
            # Control flow changes can be beneficial
            return random.uniform(0.0, 0.06)
        
        return 0.0
    
    def update_prediction_model(
        self,
        mutation: TopologyMutation,
        actual_improvement: float
    ) -> None:
        """Update prediction model with actual outcomes."""
        mutation_key = f"{mutation.mutation_type.value}_{mutation.complexity_delta}"
        
        if mutation_key not in self.mutation_outcomes:
            self.mutation_outcomes[mutation_key] = []
        
        self.mutation_outcomes[mutation_key].append({
            "improvement": actual_improvement,
            "complexity_delta": mutation.complexity_delta
        })
        
        # Keep only recent outcomes (sliding window)
        if len(self.mutation_outcomes[mutation_key]) > 20:
            self.mutation_outcomes[mutation_key] = self.mutation_outcomes[mutation_key][-20:]


class ModuleLibrary:
    """Library of pre-trained and template modules."""
    
    def __init__(self):
        self.templates = {
            "analyzer": LanguageModule(
                id="analyzer",
                prompt="Analyze the following input carefully and extract key information: {input}",
                model_weights="gpt-4"
            ),
            "classifier": LanguageModule(
                id="classifier", 
                prompt="Classify the following text into appropriate categories: {input}",
                model_weights="gpt-4"
            ),
            "summarizer": LanguageModule(
                id="summarizer",
                prompt="Create a concise summary of the following content: {input}",
                model_weights="gpt-4"
            ),
            "validator": LanguageModule(
                id="validator",
                prompt="Validate and check the accuracy of the following information: {input}",
                model_weights="gpt-4"
            ),
            "enhancer": LanguageModule(
                id="enhancer",
                prompt="Improve and enhance the following content while maintaining accuracy: {input}",
                model_weights="gpt-4"
            )
        }
    
    def get_random_module(self, existing_modules: List[str]) -> LanguageModule:
        """Get a random module template that doesn't conflict with existing ones."""
        available_templates = [
            name for name in self.templates.keys()
            if name not in existing_modules
        ]
        
        if not available_templates:
            # Create a generic module with unique ID
            module_id = f"module_{random.randint(1000, 9999)}"
            return LanguageModule(
                id=module_id,
                prompt=f"Process the following input appropriately: {{input}}",
                model_weights="gpt-4"
            )
        
        template_name = random.choice(available_templates)
        template = self.templates[template_name]
        
        # Create unique copy
        return LanguageModule(
            id=f"{template.id}_{random.randint(100, 999)}",
            prompt=template.prompt,
            model_weights=template.model_weights
        )


class NEATSystemEvolver:
    """
    Advanced NEAT-inspired system that dynamically evolves architectures.
    
    Uses predictive modeling to decide when topology evolution is beneficial
    and applies sophisticated mutations with complexity management.
    """
    
    def __init__(self, config: TopologyConfig):
        self.config = config
        self.complexity_regulator = ComplexityRegulator(config.max_complexity_threshold)
        self.performance_predictor = ArchitecturePerformancePredictor()
        self.module_library = ModuleLibrary()
        
        # Evolution tracking
        self.generation_counter = 0
        self.mutation_history = []
        self.performance_history = []
        
        logger.info("NEATSystemEvolver initialized with predictive topology evolution")
    
    def should_evolve_topology(self, optimization_state: OptimizationState) -> bool:
        """
        Intelligent decision about when topology evolution is beneficial.
        """
        current_state = optimization_state
        
        # Performance plateau detection
        plateau_detected = self._detect_performance_plateau(current_state)
        
        # Complexity vs. performance analysis
        complexity_ratio = self.complexity_regulator.analyze_efficiency(current_state.system)
        
        # Resource availability
        budget_sufficient = (
            current_state.budget_remaining >= self.config.min_topology_budget
        )
        
        # Predicted improvement potential
        suggested_mutations = self.suggest_mutations(current_state.system)
        improvement_potential = self.performance_predictor.predict_gain(
            current_state.system, suggested_mutations
        )
        
        should_evolve = (
            plateau_detected and
            complexity_ratio < self.config.max_complexity_threshold and
            budget_sufficient and
            improvement_potential > self.config.min_improvement_threshold
        )
        
        logger.debug(
            f"Topology evolution decision: {should_evolve} "
            f"(plateau: {plateau_detected}, complexity: {complexity_ratio:.2f}, "
            f"budget: {budget_sufficient}, improvement: {improvement_potential:.3f})"
        )
        
        return should_evolve
    
    def _detect_performance_plateau(self, state: OptimizationState) -> bool:
        """Detect if performance has plateaued."""
        if len(state.performance_trajectory) < 5:
            return False
        
        recent_scores = state.performance_trajectory[-5:]
        score_variance = np.var(recent_scores)
        
        # Plateau if variance is low and no recent improvements
        plateau_threshold = 0.001
        no_recent_improvement = all(
            score <= state.performance_metrics.current_best + 0.001
            for score in recent_scores
        )
        
        return score_variance < plateau_threshold and no_recent_improvement
    
    def suggest_mutations(self, system: CompoundAISystem) -> List[TopologyMutation]:
        """Generate candidate topology mutations."""
        mutations = []
        
        for mutation_type_str in self.config.mutation_types:
            try:
                mutation_type = MutationType(mutation_type_str)
                mutation = self._generate_mutation(system, mutation_type)
                if mutation:
                    mutations.append(mutation)
            except ValueError:
                logger.warning(f"Unknown mutation type: {mutation_type_str}")
        
        return mutations
    
    def _generate_mutation(
        self,
        system: CompoundAISystem,
        mutation_type: MutationType
    ) -> Optional[TopologyMutation]:
        """Generate a specific type of mutation."""
        
        if mutation_type == MutationType.ADD_MODULE:
            return self._generate_add_module_mutation(system)
        elif mutation_type == MutationType.REMOVE_MODULE:
            return self._generate_remove_module_mutation(system)
        elif mutation_type == MutationType.RECONNECT:
            return self._generate_reconnect_mutation(system)
        elif mutation_type == MutationType.CONTROL_FLOW:
            return self._generate_control_flow_mutation(system)
        
        return None
    
    def _generate_add_module_mutation(self, system: CompoundAISystem) -> TopologyMutation:
        """Generate mutation to add a new module."""
        new_module = self.module_library.get_random_module(list(system.modules.keys()))
        
        return TopologyMutation(
            mutation_type=MutationType.ADD_MODULE,
            new_module=new_module,
            complexity_delta=1.0,
            predicted_improvement=0.05 if len(system.modules) < 4 else -0.02
        )
    
    def _generate_remove_module_mutation(self, system: CompoundAISystem) -> Optional[TopologyMutation]:
        """Generate mutation to remove a module."""
        if len(system.modules) <= 1:
            return None  # Don't remove if only one module
        
        # Choose a module to remove (prefer less critical ones)
        removable_modules = list(system.modules.keys())
        target_module = random.choice(removable_modules)
        
        return TopologyMutation(
            mutation_type=MutationType.REMOVE_MODULE,
            target_module=target_module,
            complexity_delta=-1.0,
            predicted_improvement=0.02 if len(system.modules) > 3 else -0.05
        )
    
    def _generate_reconnect_mutation(self, system: CompoundAISystem) -> TopologyMutation:
        """Generate mutation to change module connections."""
        return TopologyMutation(
            mutation_type=MutationType.RECONNECT,
            connection_changes={"type": "reorder", "shuffle": True},
            complexity_delta=0.0,
            predicted_improvement=random.uniform(-0.02, 0.04)
        )
    
    def _generate_control_flow_mutation(self, system: CompoundAISystem) -> TopologyMutation:
        """Generate mutation to modify control flow."""
        return TopologyMutation(
            mutation_type=MutationType.CONTROL_FLOW,
            connection_changes={"type": "flow_modification"},
            complexity_delta=0.2,
            predicted_improvement=random.uniform(0.0, 0.06)
        )
    
    async def evolve_with_constraints(
        self,
        system: CompoundAISystem,
        performance_metrics: PerformanceMetrics
    ) -> CompoundAISystem:
        """
        Evolve topology while maintaining performance and complexity constraints.
        """
        self.generation_counter += 1
        
        # Generate candidate mutations
        candidate_mutations = self.suggest_mutations(system)
        
        if not candidate_mutations:
            logger.debug("No viable mutations generated")
            return system
        
        # Filter mutations by predicted impact
        viable_mutations = []
        for mutation in candidate_mutations:
            predicted_performance = self.performance_predictor.predict_gain(
                system, [mutation]
            )
            
            # Only consider mutations with positive predicted impact
            improvement_threshold = self.config.min_improvement_threshold
            if predicted_performance > improvement_threshold:
                mutation.predicted_improvement = predicted_performance
                viable_mutations.append(mutation)
        
        if not viable_mutations:
            logger.debug("No mutations passed prediction filtering")
            return system
        
        # Select best mutation based on predicted improvement
        best_mutation = max(viable_mutations, key=lambda m: m.predicted_improvement)
        
        logger.info(
            f"Applying topology mutation: {best_mutation.mutation_type.value} "
            f"(predicted improvement: {best_mutation.predicted_improvement:.3f})"
        )
        
        # Apply mutation
        evolved_system = self._apply_mutation(system, best_mutation)
        
        # Track mutation
        self.mutation_history.append({
            "generation": self.generation_counter,
            "mutation": best_mutation,
            "original_complexity": self.complexity_regulator.calculate_complexity(system),
            "new_complexity": self.complexity_regulator.calculate_complexity(evolved_system)
        })
        
        return evolved_system
    
    def _apply_mutation(
        self,
        system: CompoundAISystem,
        mutation: TopologyMutation
    ) -> CompoundAISystem:
        """Apply a topology mutation to the system."""
        
        if mutation.mutation_type == MutationType.ADD_MODULE:
            return self._apply_add_module(system, mutation)
        elif mutation.mutation_type == MutationType.REMOVE_MODULE:
            return self._apply_remove_module(system, mutation)
        elif mutation.mutation_type == MutationType.RECONNECT:
            return self._apply_reconnect(system, mutation)
        elif mutation.mutation_type == MutationType.CONTROL_FLOW:
            return self._apply_control_flow_change(system, mutation)
        
        return system
    
    def _apply_add_module(
        self,
        system: CompoundAISystem,
        mutation: TopologyMutation
    ) -> CompoundAISystem:
        """Apply add module mutation."""
        if not mutation.new_module:
            return system
        
        # Add new module
        new_modules = system.modules.copy()
        new_modules[mutation.new_module.id] = mutation.new_module
        
        # Update control flow to include new module
        if hasattr(system.control_flow, 'module_order'):
            new_order = system.control_flow.module_order + [mutation.new_module.id]
            new_control_flow = SequentialFlow(new_order)
        else:
            new_control_flow = system.control_flow
        
        return CompoundAISystem(
            modules=new_modules,
            control_flow=new_control_flow,
            input_schema=system.input_schema,
            output_schema=system.output_schema,
            system_id=system.system_id
        )
    
    def _apply_remove_module(
        self,
        system: CompoundAISystem,
        mutation: TopologyMutation
    ) -> CompoundAISystem:
        """Apply remove module mutation."""
        if not mutation.target_module or mutation.target_module not in system.modules:
            return system
        
        # Remove module
        new_modules = {
            k: v for k, v in system.modules.items()
            if k != mutation.target_module
        }
        
        # Update control flow
        if hasattr(system.control_flow, 'module_order'):
            new_order = [
                mod for mod in system.control_flow.module_order
                if mod != mutation.target_module
            ]
            new_control_flow = SequentialFlow(new_order) if new_order else system.control_flow
        else:
            new_control_flow = system.control_flow
        
        return CompoundAISystem(
            modules=new_modules,
            control_flow=new_control_flow,
            input_schema=system.input_schema,
            output_schema=system.output_schema,
            system_id=system.system_id
        )
    
    def _apply_reconnect(
        self,
        system: CompoundAISystem,
        mutation: TopologyMutation
    ) -> CompoundAISystem:
        """Apply reconnection mutation."""
        if not hasattr(system.control_flow, 'module_order'):
            return system
        
        # Shuffle module order
        new_order = system.control_flow.module_order.copy()
        random.shuffle(new_order)
        
        new_control_flow = SequentialFlow(new_order)
        
        return CompoundAISystem(
            modules=system.modules,
            control_flow=new_control_flow,
            input_schema=system.input_schema,
            output_schema=system.output_schema,
            system_id=system.system_id
        )
    
    def _apply_control_flow_change(
        self,
        system: CompoundAISystem,
        mutation: TopologyMutation
    ) -> CompoundAISystem:
        """Apply control flow mutation."""
        # For now, just return the system as-is
        # In practice, this would implement more sophisticated control flow changes
        return system
    
    def update_performance_predictor(
        self,
        mutation: TopologyMutation,
        actual_improvement: float
    ) -> None:
        """Update performance predictor with actual outcomes."""
        self.performance_predictor.update_prediction_model(mutation, actual_improvement)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get topology evolution metrics."""
        return {
            "generation_counter": self.generation_counter,
            "mutations_applied": len(self.mutation_history),
            "mutation_types": {
                mt.value: sum(1 for m in self.mutation_history if m["mutation"].mutation_type == mt)
                for mt in MutationType
            },
            "average_complexity_change": np.mean([
                m["new_complexity"] - m["original_complexity"]
                for m in self.mutation_history
            ]) if self.mutation_history else 0.0,
            "predictor_accuracy": len(self.performance_predictor.mutation_outcomes)
        }