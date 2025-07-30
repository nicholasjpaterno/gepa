"""Algorithm 4: System Aware Merge implementation."""

import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .system import CompoundAISystem, LanguageModule
from .pareto import Candidate

# Import advanced algorithms when available
try:
    from ..algorithms.advanced.compatibility_analysis import SophisticatedCompatibilityAnalyzer, InteractionContext
    from ..algorithms.advanced.complementarity_analysis import AdvancedComplementarityAnalyzer
    from ..algorithms.advanced.desirability_scoring import AdvancedDesirabilityScorer, MergeContext
    ADVANCED_ALGORITHM4_AVAILABLE = True
except ImportError:
    ADVANCED_ALGORITHM4_AVAILABLE = False


@dataclass
class MergeAnalysis:
    """Analysis result for determining if module combination is desirable."""
    is_desirable: bool
    compatibility_score: float
    complementarity_score: float
    performance_impact: float
    rationale: str


class Algorithm4SystemAwareMerge:
    """
    Algorithm 4: System Aware Merge strategy for compound AI optimization.
    
    This implements the sophisticated crossover strategy mentioned in the paper
    that goes beyond simple random module selection by analyzing:
    1. Module compatibility and complementarity  
    2. Performance synergies between parent systems
    3. System-level coherence after merge
    """
    
    def __init__(self, config: Optional[Any] = None):
        self.merge_history = []  # Track previous merges for learning
        self.config = config
        
        # Initialize advanced algorithms if available and enabled
        self.compatibility_analyzer = None
        self.complementarity_analyzer = None
        self.desirability_scorer = None
        
        if ADVANCED_ALGORITHM4_AVAILABLE and config and hasattr(config, 'advanced'):
            if config.advanced.compatibility_analysis_depth == "deep":
                self.compatibility_analyzer = SophisticatedCompatibilityAnalyzer()
            
            if config.advanced.enable_statistical_testing:  
                self.complementarity_analyzer = AdvancedComplementarityAnalyzer()
            
            if config.advanced.enable_mcda_scoring:
                self.desirability_scorer = AdvancedDesirabilityScorer()
    
    def system_aware_merge(
        self,
        parent1: Candidate,
        parent2: Candidate,
        training_dataset: List[Dict[str, Any]],
        evaluation_scores: Optional[Dict[str, Dict[int, float]]] = None
    ) -> Optional[CompoundAISystem]:
        """
        Perform Algorithm 4: System Aware Merge between two parent candidates.
        
        Args:
            parent1: First parent candidate
            parent2: Second parent candidate  
            training_dataset: Training data for compatibility analysis
            evaluation_scores: Optional performance scores for informed merging
            
        Returns:
            New merged system or None if merge is not desirable
        """
        
        system1 = parent1.system
        system2 = parent2.system
        
        # Step 1: Check basic compatibility
        if not self._systems_compatible(system1, system2):
            return None
        
        # Step 2: Analyze module combinations for desirability
        merge_plan = self._analyze_module_combinations(
            system1, system2, parent1, parent2, evaluation_scores
        )
        
        if not merge_plan:
            return None
        
        # Step 3: Execute the merge based on analysis
        merged_system = self._execute_merge(
            system1, system2, merge_plan
        )
        
        # Step 4: Record merge for learning
        self._record_merge_history(parent1, parent2, merge_plan)
        
        return merged_system
    
    def _systems_compatible(
        self, 
        system1: CompoundAISystem, 
        system2: CompoundAISystem
    ) -> bool:
        """Check if two systems are compatible for merging."""
        
        # Systems must have the same module structure
        if set(system1.modules.keys()) != set(system2.modules.keys()):
            return False
        
        # Control flow must be compatible
        # In a full implementation, this would check control flow compatibility
        # For now, we assume sequential flows are compatible
        
        # Input/output schemas must match
        return (
            system1.input_schema.fields == system2.input_schema.fields and
            system1.output_schema.fields == system2.output_schema.fields
        )
    
    def _analyze_module_combinations(
        self,
        system1: CompoundAISystem,
        system2: CompoundAISystem, 
        parent1: Candidate,
        parent2: Candidate,
        evaluation_scores: Optional[Dict[str, Dict[int, float]]] = None
    ) -> Optional[Dict[str, str]]:
        """
        Analyze which modules to take from which parent.
        
        This implements the core logic of Algorithm 4:
        - Check if module combinations are desirable
        - Consider performance complementarity
        - Ensure system-level coherence
        
        Returns:
            Dictionary mapping module_id -> parent_id, or None if no good merge found
        """
        
        module_ids = list(system1.modules.keys())
        best_combination = None
        best_score = -float('inf')
        
        # Generate all possible combinations (2^n where n = number of modules)
        # For each module, we can choose from parent1 or parent2
        for i in range(1, 2**len(module_ids) - 1):  # Skip all-parent1 and all-parent2
            combination = {}
            
            for j, module_id in enumerate(module_ids):
                # Use bit to decide which parent
                if (i >> j) & 1:
                    combination[module_id] = parent1.id
                else:
                    combination[module_id] = parent2.id
            
            # Check if this combination is desirable
            analysis = self._is_combination_desirable(
                combination, system1, system2, parent1, parent2, evaluation_scores
            )
            
            if analysis.is_desirable and analysis.compatibility_score > best_score:
                best_score = analysis.compatibility_score
                best_combination = combination
        
        return best_combination
    
    def _is_combination_desirable(
        self,
        combination: Dict[str, str],
        system1: CompoundAISystem,
        system2: CompoundAISystem,
        parent1: Candidate, 
        parent2: Candidate,
        evaluation_scores: Optional[Dict[str, Dict[int, float]]] = None
    ) -> MergeAnalysis:
        """
        Implementation of the DESIRABLE function from Algorithm 4.
        
        Uses advanced analysis if available, otherwise falls back to heuristics.
        """
        
        # Use advanced analysis if available
        if (self.compatibility_analyzer and self.complementarity_analyzer and 
            self.desirability_scorer and evaluation_scores):
            
            try:
                return self._advanced_desirability_analysis(
                    combination, system1, system2, parent1, parent2, evaluation_scores
                )
            except Exception as e:
                # Fall back to heuristic analysis if advanced analysis fails
                pass
        
        # Fallback to original heuristic-based analysis
        return self._heuristic_desirability_analysis(
            combination, system1, system2, parent1, parent2, evaluation_scores
        )
    
    def _advanced_desirability_analysis(
        self,
        combination: Dict[str, str],
        system1: CompoundAISystem,
        system2: CompoundAISystem,
        parent1: Candidate,
        parent2: Candidate,
        evaluation_scores: Dict[str, Dict[int, float]]
    ) -> MergeAnalysis:
        """Advanced desirability analysis using sophisticated algorithms."""
        
        # Create interaction context
        interaction_context = InteractionContext(
            workflow_position={module_id: i for i, module_id in enumerate(combination.keys())},
            data_flow={},
            execution_history={},
            performance_metrics={}
        )
        
        # Analyze compatibility between modules
        total_compatibility = 0.0
        module_count = 0
        
        modules = list(combination.keys())
        for i, module1_id in enumerate(modules):
            for module2_id in modules[i+1:]:
                # Get modules from appropriate systems
                if combination[module1_id] == parent1.id:
                    module1 = system1.modules[module1_id]
                else:
                    module1 = system2.modules[module1_id]
                    
                if combination[module2_id] == parent1.id:
                    module2 = system1.modules[module2_id]  
                else:
                    module2 = system2.modules[module2_id]
                
                # Analyze compatibility
                compat_analysis = self.compatibility_analyzer.analyze_module_compatibility(
                    module1, module2, interaction_context
                )
                total_compatibility += compat_analysis.overall_score
                module_count += 1
        
        compatibility_score = total_compatibility / max(module_count, 1)
        
        # Analyze complementarity
        parent1_scores = evaluation_scores.get(parent1.id, {})
        parent2_scores = evaluation_scores.get(parent2.id, {})
        
        complementarity_analysis = self.complementarity_analyzer.analyze_complementarity(
            parent1_scores, parent2_scores
        )
        
        # Create merge context
        merge_context = MergeContext(
            parent1_id=parent1.id,
            parent2_id=parent2.id,
            system_complexity=len(combination),
            merge_history=getattr(self, 'merge_history', {}),
            optimization_phase="exploration",
            budget_remaining=1.0
        )
        
        # Calculate performance impact
        performance_impact = complementarity_analysis.predicted_ensemble_gain
        
        # Calculate desirability using MCDA
        desirability_result = self.desirability_scorer.calculate_desirability(
            compatibility_analysis=type('CompatAnalysis', (), {'overall_score': compatibility_score})(),
            complementarity_analysis=complementarity_analysis,
            performance_impact={'expected_gain': performance_impact},
            context=merge_context
        )
        
        return MergeAnalysis(
            is_desirable=desirability_result.is_desirable,
            compatibility_score=compatibility_score,
            complementarity_score=complementarity_analysis.complementarity_strength,
            performance_impact=performance_impact,
            rationale=desirability_result.explanation
        )
    
    def _heuristic_desirability_analysis(
        self,
        combination: Dict[str, str],
        system1: CompoundAISystem,
        system2: CompoundAISystem,
        parent1: Candidate,
        parent2: Candidate,
        evaluation_scores: Optional[Dict[str, Dict[int, float]]] = None
    ) -> MergeAnalysis:
        """Original heuristic-based desirability analysis."""
        
        compatibility_score = 0.0
        complementarity_score = 0.0
        performance_impact = 0.0
        rationale_parts = []
        
        # Check each module combination
        for module_id in combination:
            parent_id = combination[module_id]
            
            if parent_id == parent1.id:
                selected_system = system1
                other_system = system2
                selected_parent = parent1
                other_parent = parent2
            else:
                selected_system = system2
                other_system = system1
                selected_parent = parent2
                other_parent = parent1
            
            # Analyze compatibility with other selected modules
            module_compat = self._analyze_module_compatibility(
                module_id, selected_system, combination
            )
            compatibility_score += module_compat
            
            # Analyze performance complementarity
            if evaluation_scores:
                perf_complement = self._analyze_performance_complementarity(
                    module_id, selected_parent, other_parent, evaluation_scores
                )
                complementarity_score += perf_complement
            
            # Check if this module choice improves overall performance
            perf_impact = self._estimate_performance_impact(
                module_id, selected_parent, other_parent
            )
            performance_impact += perf_impact
        
        # Overall desirability score
        total_score = (
            compatibility_score + 
            complementarity_score + 
            performance_impact
        ) / len(combination)
        
        # Determine if combination is desirable
        is_desirable = (
            total_score > 0.0 and  # Must be positive improvement
            compatibility_score > -0.5 and  # Not too incompatible
            len(set(combination.values())) == 2  # Must use both parents
        )
        
        if is_desirable:
            rationale_parts.append(f"Good overall score: {total_score:.2f}")
        if compatibility_score > 0:
            rationale_parts.append("Modules are compatible")
        if complementarity_score > 0:
            rationale_parts.append("Performance complementarity detected")
        
        return MergeAnalysis(
            is_desirable=is_desirable,
            compatibility_score=total_score,
            complementarity_score=complementarity_score,
            performance_impact=performance_impact,
            rationale="; ".join(rationale_parts)
        )
    
    def _analyze_module_compatibility(
        self,
        module_id: str,
        selected_system: CompoundAISystem,
        combination: Dict[str, str]
    ) -> float:
        """
        Analyze compatibility between modules in the combination.
        
        This is a simplified heuristic - in practice would analyze:
        - Prompt style consistency
        - Input/output format compatibility  
        - Semantic coherence
        """
        
        # Simple heuristic: modules from same system are more compatible
        same_system_count = sum(
            1 for other_module, parent_id in combination.items() 
            if other_module != module_id and parent_id == combination[module_id]
        )
        
        total_other_modules = len(combination) - 1
        if total_other_modules == 0:
            return 0.0
        
        # Prefer some mixing but not complete fragmentation
        same_system_ratio = same_system_count / total_other_modules
        
        # Sweet spot around 0.3-0.7 (some mixing but not too much)
        if 0.3 <= same_system_ratio <= 0.7:
            return 0.5
        elif same_system_ratio < 0.3:
            return -0.2  # Too fragmented
        else:
            return 0.2   # Too uniform
    
    def _analyze_performance_complementarity(
        self,
        module_id: str,
        selected_parent: Candidate,
        other_parent: Candidate,
        evaluation_scores: Dict[str, Dict[int, float]]
    ) -> float:
        """
        Analyze if selected module complements other modules performance-wise.
        """
        
        if (selected_parent.id not in evaluation_scores or 
            other_parent.id not in evaluation_scores):
            return 0.0
        
        selected_scores = evaluation_scores[selected_parent.id]
        other_scores = evaluation_scores[other_parent.id]
        
        # Simple heuristic: if selected parent performs better on some instances
        # where other parent performs worse, that's complementary
        complementarity = 0.0
        count = 0
        
        for instance_id in selected_scores:
            if instance_id in other_scores:
                selected_score = selected_scores[instance_id]
                other_score = other_scores[instance_id]
                
                # If selected is better on this instance, that's good
                if selected_score > other_score:
                    complementarity += (selected_score - other_score)
                count += 1
        
        return complementarity / max(count, 1)
    
    def _estimate_performance_impact(
        self,
        module_id: str,
        selected_parent: Candidate,
        other_parent: Candidate  
    ) -> float:
        """
        Estimate performance impact of selecting this module.
        
        Simple heuristic based on parent performance.
        """
        
        if not selected_parent.scores or not other_parent.scores:
            return 0.0
        
        selected_avg = sum(selected_parent.scores.values()) / len(selected_parent.scores)
        other_avg = sum(other_parent.scores.values()) / len(other_parent.scores)
        
        # Prefer modules from better-performing parent
        return (selected_avg - other_avg) * 0.1
    
    def _execute_merge(
        self,
        system1: CompoundAISystem,
        system2: CompoundAISystem,
        merge_plan: Dict[str, str]
    ) -> CompoundAISystem:
        """
        Execute the merge based on the merge plan.
        """
        
        new_modules = {}
        
        for module_id, parent_id in merge_plan.items():
            if parent_id == "parent1":
                new_modules[module_id] = system1.modules[module_id]
            else:
                new_modules[module_id] = system2.modules[module_id]
        
        # Create merged system
        import uuid
        merged_system = CompoundAISystem(
            modules=new_modules,
            control_flow=system1.control_flow,  # Use parent1's control flow
            input_schema=system1.input_schema,
            output_schema=system1.output_schema,
            system_id=f"merge_{uuid.uuid4().hex[:8]}"
        )
        
        return merged_system
    
    def _record_merge_history(
        self,
        parent1: Candidate,
        parent2: Candidate,
        merge_plan: Dict[str, str]
    ) -> None:
        """
        Record merge history for learning and improvement.
        """
        
        merge_record = {
            "parent1_id": parent1.id,
            "parent2_id": parent2.id,
            "parent1_scores": parent1.scores,
            "parent2_scores": parent2.scores,
            "merge_plan": merge_plan,
            "timestamp": "now"  # Would use actual timestamp
        }
        
        self.merge_history.append(merge_record)
        
        # Keep history bounded
        if len(self.merge_history) > 100:
            self.merge_history = self.merge_history[-100:]
    
    def get_merge_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about merge operations.
        """
        
        if not self.merge_history:
            return {"total_merges": 0}
        
        stats = {
            "total_merges": len(self.merge_history),
            "unique_parent_pairs": len(set(
                (record["parent1_id"], record["parent2_id"]) 
                for record in self.merge_history
            ))
        }
        
        return stats