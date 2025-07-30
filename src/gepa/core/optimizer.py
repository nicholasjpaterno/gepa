"""Main GEPA optimizer implementing Algorithm 1 from the paper."""

import uuid
import asyncio
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import structlog

from ..config import GEPAConfig
from ..inference.base import InferenceClient
from ..inference.factory import InferenceFactory
from .system import CompoundAISystem
from .pareto import ParetoFrontier, Candidate
from .mutation import ReflectiveMutator, MutationType, Trajectory
from .algorithm4 import Algorithm4SystemAwareMerge
from ..evaluation.base import Evaluator


@dataclass
class OptimizationResult:
    """Result of GEPA optimization."""
    best_system: CompoundAISystem
    best_score: float
    pareto_frontier: ParetoFrontier
    total_rollouts: int
    total_cost: float
    optimization_history: List[Dict[str, Any]]


class GEPAOptimizer:
    """
    GEPA (GeneticPareto) optimizer implementing Algorithm 1.
    
    Uses reflective prompt evolution with Pareto-based candidate sampling
    to optimize compound AI systems efficiently.
    """
    
    def __init__(
        self,
        config: GEPAConfig,
        evaluator: Evaluator,
        inference_client: Optional[InferenceClient] = None,
        reflection_client: Optional[InferenceClient] = None
    ):
        self.config = config
        self.evaluator = evaluator
        
        # Create inference clients
        if inference_client is None:
            self.inference_client = InferenceFactory.create_client(config.inference)
        else:
            self.inference_client = inference_client
            
        if reflection_client is None:
            reflection_config = config.inference
            if config.optimization.reflection_model:
                reflection_config = reflection_config.copy()
                reflection_config.model = config.optimization.reflection_model
            self.reflection_client = InferenceFactory.create_client(reflection_config)
        else:
            self.reflection_client = reflection_client
        
        # Initialize components
        self.pareto_frontier = ParetoFrontier(config.optimization.pareto_set_size, config)
        self.mutator = ReflectiveMutator(self.reflection_client, config)
        self.system_aware_merge = Algorithm4SystemAwareMerge(config)
        
        # Tracking
        self.rollouts_used = 0
        self.total_cost = 0.0
        self.optimization_history = []
        
        # Logger
        self.logger = structlog.get_logger(__name__)
    
    async def optimize(
        self,
        system: CompoundAISystem,
        dataset: List[Dict[str, Any]],
        max_generations: Optional[int] = None
    ) -> OptimizationResult:
        """
        Optimize a compound AI system using GEPA algorithm.
        
        Args:
            system: Initial compound AI system to optimize
            dataset: Training dataset for evaluation
            max_generations: Maximum generations (overrides budget if set)
        
        Returns:
            OptimizationResult with best system and metrics
        """
        self.logger.info(
            "Starting GEPA optimization",
            system_id=system.system_id,
            dataset_size=len(dataset),
            budget=self.config.optimization.budget
        )
        
        # Split dataset as per Algorithm 1
        random.shuffle(dataset)
        pareto_size = min(self.config.optimization.pareto_set_size, len(dataset))
        d_pareto = dataset[:pareto_size]
        d_feedback = dataset[pareto_size:]
        
        # Initialize with original system (Line 2 in Algorithm 1)
        initial_candidate = await self._evaluate_system(system, d_pareto)
        if initial_candidate:
            self.pareto_frontier.add_candidate(initial_candidate)
            self.logger.info("Added initial candidate", score=initial_candidate.scores)
        
        generation = 0
        while (self.rollouts_used < self.config.optimization.budget and 
               (max_generations is None or generation < max_generations)):
            
            generation += 1
            self.logger.info(
                "Starting generation",
                generation=generation,
                rollouts_used=self.rollouts_used,
                frontier_size=self.pareto_frontier.size()
            )
            
            # Sample candidate from Pareto frontier using Algorithm 2 if enabled
            if (hasattr(self.config, 'advanced') and 
                hasattr(self.config.advanced, 'enable_score_prediction') and
                self.config.advanced.enable_score_prediction):
                parent_candidate = self.pareto_frontier.sample_candidate_algorithm2(
                    d_pareto,  # Training dataset for Algorithm 2
                    None  # Could pass scores matrix in full implementation
                )
            else:
                parent_candidate = self.pareto_frontier.sample_candidate()
            if parent_candidate is None:
                self.logger.warning("No candidates in Pareto frontier")
                break
            
            # Generate new candidate through mutation or crossover
            new_system = await self._generate_new_candidate(
                parent_candidate, d_feedback
            )
            
            if new_system is None:
                continue
            
            # Quick evaluation on minibatch
            minibatch_size = min(
                self.config.optimization.minibatch_size,
                len(d_feedback)
            )
            # Use strategic minibatch sampling if available and enabled
            try:
                if (hasattr(self.config, 'advanced') and 
                    self.config.advanced.minibatch_strategy == "strategic"):
                    from ..algorithms.advanced.strategic_sampling import StrategicMinibatchSampler, SamplingContext
                    
                    if not hasattr(self, '_strategic_sampler'):
                        self._strategic_sampler = StrategicMinibatchSampler()
                    
                    context = SamplingContext(
                        current_candidate=parent_candidate,
                        sampling_strategy="balanced",
                        exploration_factor=0.3
                    )
                    minibatch = self._strategic_sampler.sample_minibatch(d_feedback, minibatch_size, context)
                else:
                    minibatch = random.sample(d_feedback, minibatch_size)
            except ImportError:
                minibatch = random.sample(d_feedback, minibatch_size)
            
            # Evaluate on minibatch first
            candidate = await self._evaluate_system(new_system, minibatch)
            if candidate is None:
                continue
            
            candidate.parent_id = parent_candidate.id
            candidate.generation = generation
            
            # If promising, evaluate on full Pareto set
            if self._is_promising_candidate(candidate, parent_candidate):
                full_candidate = await self._evaluate_system(new_system, d_pareto)
                if full_candidate:
                    full_candidate.parent_id = parent_candidate.id
                    full_candidate.generation = generation
                    
                    # Add to Pareto frontier
                    added = self.pareto_frontier.add_candidate(full_candidate)
                    
                    self.logger.info(
                        "Evaluated candidate",
                        candidate_id=full_candidate.id,
                        scores=full_candidate.scores,
                        cost=full_candidate.cost,
                        added_to_frontier=added
                    )
                    
                    # Record in history
                    self.optimization_history.append({
                        "generation": generation,
                        "candidate_id": full_candidate.id,
                        "parent_id": parent_candidate.id,
                        "scores": full_candidate.scores,
                        "cost": full_candidate.cost,
                        "added_to_frontier": added,
                        "rollouts_used": self.rollouts_used
                    })
            
            # Check if we should stop early
            if self._should_stop_early():
                self.logger.info("Early stopping triggered")
                break
        
        # Get best system from Pareto frontier
        best_candidate = self._select_best_candidate()
        
        result = OptimizationResult(
            best_system=best_candidate.system if best_candidate else system,
            best_score=max(best_candidate.scores.values()) if best_candidate else 0.0,
            pareto_frontier=self.pareto_frontier,
            total_rollouts=self.rollouts_used,
            total_cost=self.total_cost,
            optimization_history=self.optimization_history
        )
        
        self.logger.info(
            "Optimization completed",
            generations=generation,
            rollouts_used=self.rollouts_used,
            total_cost=self.total_cost,
            frontier_size=self.pareto_frontier.size()
        )
        
        return result
    
    async def _evaluate_system(
        self,
        system: CompoundAISystem,
        dataset: List[Dict[str, Any]]
    ) -> Optional[Candidate]:
        """Evaluate a system on a dataset."""
        if self.rollouts_used >= self.config.optimization.budget:
            return None
        
        try:
            # Execute system on dataset
            trajectories = []
            scores = {}
            total_cost = 0.0
            total_tokens = 0
            
            for data_point in dataset:
                if self.rollouts_used >= self.config.optimization.budget:
                    break
                
                # Execute system
                trajectory = await self._execute_system_with_tracking(system, data_point)
                trajectories.append(trajectory)
                
                # Update tracking
                self.rollouts_used += 1
                if hasattr(self.inference_client, 'get_metrics'):
                    metrics = self.inference_client.get_metrics()
                    total_cost += metrics.total_cost
                    total_tokens += metrics.total_tokens
            
            # Evaluate trajectories
            evaluation_result = await self.evaluator.evaluate_batch(
                [t.output_data for t in trajectories],
                [{"expected": dp.get("expected")} for dp in dataset[:len(trajectories)]]
            )
            
            scores = evaluation_result.scores
            total_cost += evaluation_result.cost
            
            # Create candidate
            candidate = Candidate(
                id=str(uuid.uuid4()),
                system=system,
                scores=scores,
                cost=total_cost,
                tokens_used=total_tokens
            )
            
            self.total_cost += total_cost
            
            return candidate
            
        except Exception as e:
            self.logger.error("Error evaluating system", error=str(e))
            return None
    
    async def _execute_system_with_tracking(
        self,
        system: CompoundAISystem,
        data_point: Dict[str, Any]
    ) -> Trajectory:
        """Execute system and track the trajectory."""
        import time
        
        start_time = time.time()
        success = False
        error = None
        output_data = {}
        
        try:
            output_data = await system.execute(data_point, self.inference_client)
            success = True
        except Exception as e:
            error = str(e)
            self.logger.error("System execution failed", error=error)
        
        total_latency = time.time() - start_time
        
        # For now, create a simplified trajectory
        # In a full implementation, you'd track each module step
        trajectory = Trajectory(
            system_id=system.system_id,
            input_data=data_point,
            output_data=output_data,
            steps=[],  # Would be populated with actual steps
            total_latency=total_latency,
            success=success,
            error=error
        )
        
        return trajectory
    
    async def _generate_new_candidate(
        self,
        parent_candidate: Candidate,
        feedback_dataset: List[Dict[str, Any]]
    ) -> Optional[CompoundAISystem]:
        """Generate new candidate through mutation or crossover."""
        
        # Decide between mutation and crossover
        if (self.config.optimization.enable_crossover and 
            random.random() < self.config.optimization.crossover_probability and
            self.pareto_frontier.size() > 1):
            return await self._perform_crossover(parent_candidate)
        else:
            return await self._perform_mutation(parent_candidate, feedback_dataset)
    
    async def _perform_mutation(
        self,
        parent_candidate: Candidate,
        feedback_dataset: List[Dict[str, Any]]
    ) -> Optional[CompoundAISystem]:
        """Perform reflective mutation on a candidate."""
        
        parent_system = parent_candidate.system
        
        # Choose a random module to mutate
        module_ids = list(parent_system.modules.keys())
        if not module_ids:
            return None
        
        module_id = random.choice(module_ids)
        
        # Choose mutation type
        mutation_types = [MutationType(t) for t in self.config.optimization.mutation_types]
        mutation_type = random.choice(mutation_types)
        
        # Get recent trajectories for reflection
        # For now, create dummy trajectories - in full implementation,
        # these would be retrieved from trajectory storage
        trajectories = []
        
        try:
            # Use Algorithm 3 if enabled, otherwise use standard mutation
            if (hasattr(self.config, 'advanced') and 
                hasattr(self.config.advanced, 'module_selection_strategy') and
                self.config.advanced.module_selection_strategy == "intelligent"):
                new_system = await self.mutator.algorithm3_reflective_mutation(
                    parent_system,
                    feedback_dataset,
                    self.inference_client,
                    self.evaluator
                )
                
                if new_system:
                    self.logger.info(
                        "Performed Algorithm 3 reflective mutation",
                        parent_id=parent_candidate.id
                    )
                    return new_system
                else:
                    # Fallback to standard mutation
                    new_prompt = await self.mutator.mutate_prompt(
                        parent_system,
                        module_id,
                        trajectories,
                        mutation_type
                    )
            else:
                # Standard mutation
                new_prompt = await self.mutator.mutate_prompt(
                    parent_system,
                    module_id,
                    trajectories,
                    mutation_type
                )
            
            # Create new system with mutated prompt
            new_system = parent_system.update_module(module_id, new_prompt)
            
            self.logger.info(
                "Performed mutation",
                module_id=module_id,
                mutation_type=mutation_type.value,
                parent_id=parent_candidate.id
            )
            
            return new_system
            
        except Exception as e:
            self.logger.error("Mutation failed", error=str(e))
            return None
    
    async def _perform_crossover(
        self,
        parent_candidate: Candidate
    ) -> Optional[CompoundAISystem]:
        """Perform crossover between two candidates using Algorithm 4: System Aware Merge."""
        
        # Select second parent from frontier
        other_candidates = [
            c for c in self.pareto_frontier.candidates 
            if c.id != parent_candidate.id
        ]
        
        if not other_candidates:
            return None
        
        second_parent = random.choice(other_candidates)
        
        # Try Algorithm 4: System Aware Merge first
        if (hasattr(self.config, 'advanced') and 
            hasattr(self.config.advanced, 'enable_mcda_scoring') and
            self.config.advanced.enable_mcda_scoring):
            new_system = self.system_aware_merge.system_aware_merge(
                parent_candidate,
                second_parent,
                [],  # Would pass training dataset in full implementation
                None  # Would pass evaluation scores in full implementation
            )
            
            if new_system:
                self.logger.info(
                    "Performed Algorithm 4 system aware merge",
                    parent1_id=parent_candidate.id,
                    parent2_id=second_parent.id
                )
                return new_system
        
        # Fallback to simple crossover if Algorithm 4 fails
        parent1_system = parent_candidate.system
        parent2_system = second_parent.system
        
        # Only crossover if systems have same modules
        if set(parent1_system.modules.keys()) != set(parent2_system.modules.keys()):
            return None
        
        new_modules = {}
        for module_id in parent1_system.modules.keys():
            # Randomly choose from which parent to take the module
            if random.random() < 0.5:
                new_modules[module_id] = parent1_system.modules[module_id]
            else:
                new_modules[module_id] = parent2_system.modules[module_id]
        
        # Create new system
        new_system = CompoundAISystem(
            modules=new_modules,
            control_flow=parent1_system.control_flow,
            input_schema=parent1_system.input_schema,
            output_schema=parent1_system.output_schema,
            system_id=f"crossover_{uuid.uuid4().hex[:8]}"
        )
        
        self.logger.info(
            "Performed simple crossover",
            parent1_id=parent_candidate.id,
            parent2_id=second_parent.id
        )
        
        return new_system
    
    def _is_promising_candidate(
        self,
        candidate: Candidate,
        parent: Candidate
    ) -> bool:
        """Check if candidate is promising enough for full evaluation."""
        
        # If no scores, consider promising
        if not candidate.scores or not parent.scores:
            return True
        
        # Compare average scores
        candidate_avg = sum(candidate.scores.values()) / len(candidate.scores)
        parent_avg = sum(parent.scores.values()) / len(parent.scores)
        
        # Promising if better than parent or within threshold
        improvement_threshold = 0.05  # 5% worse is still acceptable
        return candidate_avg >= parent_avg - improvement_threshold
    
    def _should_stop_early(self) -> bool:
        """Check if optimization should stop early."""
        
        # Stop if no improvement in last N generations
        if len(self.optimization_history) < 10:
            return False
        
        recent_history = self.optimization_history[-10:]
        improvements = sum(1 for h in recent_history if h["added_to_frontier"])
        
        # No improvements in last 10 generations
        return improvements == 0
    
    def _select_best_candidate(self) -> Optional[Candidate]:
        """Select the best candidate from Pareto frontier."""
        if self.pareto_frontier.is_empty():
            return None
        
        # Select candidate with highest average score
        best_candidate = None
        best_score = -float('inf')
        
        for candidate in self.pareto_frontier.candidates:
            if candidate.scores:
                avg_score = sum(candidate.scores.values()) / len(candidate.scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_candidate = candidate
        
        return best_candidate or self.pareto_frontier.candidates[0]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = {
            "rollouts_used": self.rollouts_used,
            "total_cost": self.total_cost,
            "pareto_frontier": self.pareto_frontier.get_statistics(),
            "generations": len(set(h["generation"] for h in self.optimization_history)),
            "successful_mutations": sum(
                1 for h in self.optimization_history if h["added_to_frontier"]
            )
        }
        
        return stats