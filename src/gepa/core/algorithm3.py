"""Algorithm 3: Reflective Prompt Mutation implementation."""

import asyncio
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..inference.base import InferenceClient, InferenceRequest
from .system import CompoundAISystem
from .mutation import Trajectory, TrajectoryStep


class ReflectionStrategy(Enum):
    """Different reflection strategies for prompt mutation."""
    FAILURE_ANALYSIS = "failure_analysis"
    SUCCESS_ANALYSIS = "success_analysis" 
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    ROBUSTNESS_IMPROVEMENT = "robustness_improvement"


@dataclass
class ReflectionContext:
    """Context for reflective prompt mutation."""
    target_module_id: str
    trajectories: List[Trajectory]
    evaluation_traces: List[Dict[str, Any]]
    current_prompt: str
    system: CompoundAISystem
    strategy: ReflectionStrategy


class Algorithm3ReflectiveMutation:
    """
    Algorithm 3: Reflective Prompt Mutation as described in the paper.
    
    This class implements the exact algorithm described in Section 3.2:
    1. Select target module via round-robin
    2. Generate rollouts on minibatch
    3. Extract execution traces
    4. Use LLM reflection to identify issues and propose improvements
    5. Generate new prompt based on reflection
    """
    
    # Meta-prompt template based on Appendix B from the paper
    REFLECTION_META_PROMPT = """# System Execution Analysis and Prompt Improvement

## System Overview
You are analyzing the execution of a compound AI system module to improve its prompt.

## Module Information
**Module ID**: {module_id}
**Current Prompt**: 
```
{current_prompt}
```

## Execution Analysis

### Successful Executions
{successful_executions}

### Failed Executions  
{failed_executions}

### Evaluation Traces
{evaluation_traces}

## Performance Metrics
- Success Rate: {success_rate:.1%}
- Average Latency: {avg_latency:.2f}s
- Error Patterns: {error_patterns}

## Reflection Task

Based on the execution traces and evaluation feedback above, analyze the current prompt and identify specific issues that led to failures or suboptimal performance. Consider:

1. **Clarity Issues**: Are the instructions clear and unambiguous?
2. **Missing Constraints**: Are there important constraints or edge cases not addressed?
3. **Format Issues**: Does the prompt specify the required output format clearly?
4. **Context Issues**: Does the prompt provide sufficient context for the task?
5. **Reasoning Issues**: Does the prompt guide the model through proper reasoning steps?

## Improved Prompt

Provide a revised prompt that addresses the identified issues. The new prompt should:
- Fix specific problems identified in the failure cases
- Maintain successful behaviors from working cases
- Be clear, specific, and actionable
- Include necessary examples or constraints

**New Prompt**:
```
[Your improved prompt here]
```

## Explanation of Changes
Briefly explain the key changes made and why they address the identified issues:
"""
    
    def __init__(
        self, 
        reflection_client: InferenceClient,
        max_rollouts_per_reflection: int = 5,
        minibatch_size: int = 3
    ):
        self.reflection_client = reflection_client
        self.max_rollouts_per_reflection = max_rollouts_per_reflection
        self.minibatch_size = minibatch_size
        self.module_selection_counter = {}  # For round-robin selection
    
    async def reflective_mutation(
        self,
        system: CompoundAISystem,
        training_dataset: List[Dict[str, Any]],
        inference_client: InferenceClient,
        evaluator: Any,  # Evaluator interface
        strategy: ReflectionStrategy = ReflectionStrategy.FAILURE_ANALYSIS
    ) -> Optional[CompoundAISystem]:
        """
        Perform Algorithm 3: Reflective Prompt Mutation.
        
        Args:
            system: The compound AI system to mutate
            training_dataset: Training data for generating rollouts
            inference_client: Client for system execution
            evaluator: Evaluator for assessing performance
            strategy: Reflection strategy to use
            
        Returns:
            New system with mutated prompt, or None if mutation failed
        """
        
        # Step 1: Select target module (round-robin as mentioned in paper)
        target_module_id = self._select_target_module(system)
        if not target_module_id:
            return None
        
        # Step 2: Sample minibatch from training dataset
        minibatch = random.sample(
            training_dataset, 
            min(self.minibatch_size, len(training_dataset))
        )
        
        # Step 3: Generate rollouts and collect traces
        trajectories = []
        evaluation_traces = []
        
        for data_point in minibatch:
            try:
                # Execute system and capture trajectory
                trajectory = await self._execute_with_tracing(
                    system, data_point, inference_client
                )
                trajectories.append(trajectory)
                
                # Get evaluation trace (rich diagnostic information)
                eval_trace = await self._get_evaluation_trace(
                    trajectory, data_point, evaluator
                )
                evaluation_traces.append(eval_trace)
                
            except Exception as e:
                # Create error trajectory
                error_trajectory = Trajectory(
                    system_id=system.system_id,
                    input_data=data_point,
                    output_data={},
                    steps=[],
                    total_latency=0.0,
                    success=False,
                    error=str(e)
                )
                trajectories.append(error_trajectory)
                evaluation_traces.append({"error": str(e), "success": False})
        
        # Step 4: Create reflection context
        reflection_context = ReflectionContext(
            target_module_id=target_module_id,
            trajectories=trajectories,
            evaluation_traces=evaluation_traces,
            current_prompt=system.modules[target_module_id].prompt,
            system=system,
            strategy=strategy
        )
        
        # Step 5: Perform LLM-based reflection
        new_prompt = await self._perform_reflection(reflection_context)
        if not new_prompt:
            return None
        
        # Step 6: Create new system with updated prompt
        new_system = system.update_module(target_module_id, new_prompt)
        
        return new_system
    
    def _select_target_module(self, system: CompoundAISystem) -> Optional[str]:
        """Select target module using round-robin strategy."""
        module_ids = list(system.modules.keys())
        if not module_ids:
            return None
        
        # Initialize counters for new modules
        for module_id in module_ids:
            if module_id not in self.module_selection_counter:
                self.module_selection_counter[module_id] = 0
        
        # Select module with lowest counter (round-robin)
        selected_module = min(module_ids, key=lambda x: self.module_selection_counter[x])
        self.module_selection_counter[selected_module] += 1
        
        return selected_module
    
    async def _execute_with_tracing(
        self,
        system: CompoundAISystem,
        data_point: Dict[str, Any],
        inference_client: InferenceClient
    ) -> Trajectory:
        """Execute system with detailed tracing of each module."""
        import time
        
        start_time = time.time()
        steps = []
        success = False
        error = None
        output_data = {}
        
        try:
            # For now, simulate detailed tracing
            # In a full implementation, this would instrument each module execution
            output_data = await system.execute(data_point, inference_client)
            
            # Create synthetic steps for each module (would be real in full implementation)
            for module_id, module in system.modules.items():
                step = TrajectoryStep(
                    module_id=module_id,
                    input_data=data_point,
                    output_data=output_data,
                    prompt_used=module.prompt,
                    latency=0.1,  # Would be actual latency
                    error=None
                )
                steps.append(step)
            
            success = True
            
        except Exception as e:
            error = str(e)
        
        total_latency = time.time() - start_time
        
        return Trajectory(
            system_id=system.system_id,
            input_data=data_point,
            output_data=output_data,
            steps=steps,
            total_latency=total_latency,
            success=success,
            error=error
        )
    
    async def _get_evaluation_trace(
        self,
        trajectory: Trajectory,
        data_point: Dict[str, Any],
        evaluator: Any
    ) -> Dict[str, Any]:
        """Get rich evaluation trace with diagnostic information."""
        try:
            # Get evaluation result with detailed feedback
            result = await evaluator.evaluate_single(
                trajectory.output_data,
                {"expected": data_point.get("expected", {})}
            )
            
            return {
                "success": trajectory.success,
                "scores": getattr(result, 'scores', {}),
                "feedback": getattr(result, 'feedback', ""),
                "detailed_analysis": getattr(result, 'detailed_analysis', {}),
                "execution_steps": len(trajectory.steps),
                "total_latency": trajectory.total_latency
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_steps": len(trajectory.steps),
                "total_latency": trajectory.total_latency
            }
    
    async def _perform_reflection(
        self, 
        context: ReflectionContext
    ) -> Optional[str]:
        """Perform LLM-based reflection to generate improved prompt."""
        
        # Analyze trajectories
        successful_trajectories = [t for t in context.trajectories if t.success]
        failed_trajectories = [t for t in context.trajectories if not t.success]
        
        # Calculate metrics
        success_rate = len(successful_trajectories) / max(len(context.trajectories), 1)
        avg_latency = sum(t.total_latency for t in context.trajectories) / max(len(context.trajectories), 1)
        
        # Extract error patterns
        error_patterns = self._extract_error_patterns(failed_trajectories)
        
        # Format execution examples
        successful_executions = self._format_executions(successful_trajectories[:2])
        failed_executions = self._format_executions(failed_trajectories[:2])
        evaluation_traces = self._format_evaluation_traces(context.evaluation_traces[:3])
        
        # Create reflection prompt
        reflection_prompt = self.REFLECTION_META_PROMPT.format(
            module_id=context.target_module_id,
            current_prompt=context.current_prompt,
            successful_executions=successful_executions,
            failed_executions=failed_executions,
            evaluation_traces=evaluation_traces,
            success_rate=success_rate,
            avg_latency=avg_latency,
            error_patterns=error_patterns
        )
        
        # Generate reflection
        request = InferenceRequest(
            prompt=reflection_prompt,
            max_tokens=2048,
            temperature=0.7
        )
        
        try:
            response = await self.reflection_client.generate(request)
            
            # Extract new prompt from response
            new_prompt = self._extract_prompt_from_reflection(response.text)
            
            return new_prompt
            
        except Exception as e:
            print(f"Reflection failed: {e}")
            return None
    
    def _extract_error_patterns(self, failed_trajectories: List[Trajectory]) -> str:
        """Extract common error patterns from failed trajectories."""
        if not failed_trajectories:
            return "No failures to analyze"
        
        error_counts = {}
        for trajectory in failed_trajectories:
            if trajectory.error:
                error_type = trajectory.error.split(':')[0].strip()
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        if not error_counts:
            return "No specific error patterns identified"
        
        patterns = []
        for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            patterns.append(f"{error_type} ({count}x)")
        
        return ", ".join(patterns[:3])  # Top 3 patterns
    
    def _format_executions(self, trajectories: List[Trajectory]) -> str:
        """Format trajectory executions for the reflection prompt."""
        if not trajectories:
            return "None available"
        
        formatted = []
        for i, trajectory in enumerate(trajectories):
            formatted.append(f"""
**Example {i+1}:**
- Input: {trajectory.input_data}
- Output: {trajectory.output_data}
- Success: {trajectory.success}
- Latency: {trajectory.total_latency:.2f}s
{f"- Error: {trajectory.error}" if trajectory.error else ""}
""")
        
        return "".join(formatted)
    
    def _format_evaluation_traces(self, traces: List[Dict[str, Any]]) -> str:
        """Format evaluation traces for the reflection prompt."""
        if not traces:
            return "No evaluation traces available"
        
        formatted = []
        for i, trace in enumerate(traces):
            formatted.append(f"""
**Trace {i+1}:**
- Success: {trace.get('success', False)}
- Scores: {trace.get('scores', {})}
- Feedback: {trace.get('feedback', 'No feedback')}
""")
        
        return "".join(formatted)
    
    def _extract_prompt_from_reflection(self, reflection_text: str) -> Optional[str]:
        """Extract the new prompt from reflection response."""
        # Look for prompt in code blocks
        import re
        
        # Try to find content between ```
        code_block_pattern = r'```(?:.*?\n)?(.*?)```'
        matches = re.findall(code_block_pattern, reflection_text, re.DOTALL)
        
        if matches:
            # Return the last code block (likely the new prompt)
            return matches[-1].strip()
        
        # Fallback: look for "New Prompt:" section
        lines = reflection_text.split('\n')
        collecting = False
        prompt_lines = []
        
        for line in lines:
            if '**New Prompt**' in line or 'New Prompt:' in line:
                collecting = True
                continue
            elif collecting:
                if line.strip().startswith('##') or line.strip().startswith('**'):
                    break
                prompt_lines.append(line)
        
        if prompt_lines:
            return '\n'.join(prompt_lines).strip()
        
        return None