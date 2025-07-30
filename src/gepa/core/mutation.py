"""Reflective mutation operations for GEPA."""

import random
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from jinja2 import Template

from ..inference.base import InferenceClient, InferenceRequest
from .system import CompoundAISystem


class MutationType(Enum):
    """Types of mutations available."""
    REWRITE = "rewrite"
    INSERT = "insert" 
    DELETE = "delete"
    COMPRESS = "compress"


@dataclass
class TrajectoryStep:
    """A single step in a system trajectory."""
    module_id: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    prompt_used: str
    latency: float
    error: Optional[str] = None


@dataclass
class Trajectory:
    """Complete system trajectory."""
    system_id: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    steps: List[TrajectoryStep]
    total_latency: float
    success: bool
    error: Optional[str] = None


class ReflectiveMutator:
    """Handles reflective mutation of prompts based on trajectory analysis.
    
    Implements Algorithm 3: Reflective Prompt Mutation from the GEPA paper.
    """
    
    REFLECTION_TEMPLATES = {
        MutationType.REWRITE: Template("""
# Prompt Analysis and Improvement

## Current Prompt
{{ current_prompt }}

## System Trajectory Analysis

### Input
{{ input_data }}

### Output
{{ output_data }}

### Issues Identified
{% for issue in issues %}
- {{ issue }}
{% endfor %}

### Performance Analysis
- Success: {{ success }}
- Total Latency: {{ total_latency }}s
{% if error %}
- Error: {{ error }}
{% endif %}

## Task
Based on the analysis above, rewrite the prompt to address the identified issues and improve performance. The new prompt should:
1. Fix any logical errors or unclear instructions
2. Improve clarity and specificity  
3. Add necessary constraints or examples
4. Maintain the original intent while enhancing effectiveness

## Improved Prompt
"""),
        
        MutationType.INSERT: Template("""
# Prompt Enhancement through Addition

## Current Prompt
{{ current_prompt }}

## Analysis
The current prompt needs additional information to handle the following scenario better:

### Issues to Address
{% for issue in issues %}
- {{ issue }}
{% endfor %}

### Successful Examples
{% for example in examples %}
Input: {{ example.input }}
Expected: {{ example.expected }}
{% endfor %}

## Task
Insert additional instructions, examples, or constraints into the existing prompt to address the issues. Add content that will help the model:
1. Handle edge cases better
2. Follow more specific guidelines
3. Produce more consistent outputs

## Enhanced Prompt
"""),
        
        MutationType.DELETE: Template("""
# Prompt Simplification

## Current Prompt
{{ current_prompt }}

## Analysis
The current prompt may be too verbose or contain unnecessary elements:

### Potential Issues
{% for issue in issues %}
- {{ issue }}
{% endfor %}

### Performance Metrics
- Current Length: {{ prompt_length }} characters
- Target Reduction: ~{{ target_reduction }}%

## Task
Remove unnecessary parts of the prompt while maintaining its core functionality. Focus on:
1. Eliminating redundant instructions
2. Removing overly complex examples
3. Simplifying language while keeping clarity
4. Maintaining essential constraints

## Simplified Prompt
"""),
        
        MutationType.COMPRESS: Template("""
# Prompt Compression

## Current Prompt
{{ current_prompt }}

## Compression Goal
Create a more concise version that maintains the same functionality with fewer tokens.

### Key Requirements to Preserve
{% for req in key_requirements %}
- {{ req }}
{% endfor %}

### Performance Constraints
- Current length: {{ current_length }} tokens
- Target length: {{ target_length }} tokens
- Must maintain: {{ performance_threshold }}% performance

## Task
Compress the prompt by:
1. Using more concise language
2. Combining related instructions
3. Using abbreviations where appropriate
4. Removing less critical details

## Compressed Prompt
""")
    }
    
    def __init__(self, reflection_client: InferenceClient, config: Optional[Any] = None):
        self.reflection_client = reflection_client
        self.config = config
        self.module_selection_counter = {}  # For round-robin module selection
        
        # Initialize tracking for intelligent selection
        self.performance_history = {}  # Module performance over time
        self.mutation_history = {}     # Mutation success/failure history
    
    async def mutate_prompt(
        self,
        system: CompoundAISystem,
        module_id: str,
        trajectories: List[Trajectory],
        mutation_type: MutationType,
        **kwargs
    ) -> str:
        """
        Mutate a prompt based on trajectory analysis.
        
        Args:
            system: The compound AI system
            module_id: ID of module to mutate
            trajectories: Recent trajectories for analysis
            mutation_type: Type of mutation to perform
            **kwargs: Additional parameters for mutation
        
        Returns:
            New mutated prompt
        """
        if module_id not in system.modules:
            raise ValueError(f"Module {module_id} not found in system")
        
        current_prompt = system.modules[module_id].prompt
        
        # Analyze trajectories to identify issues
        issues = self._analyze_trajectories(trajectories, module_id)
        
        # Prepare template context
        context = {
            "current_prompt": current_prompt,
            "issues": issues,
            "success_rate": self._calculate_success_rate(trajectories),
            "avg_latency": self._calculate_avg_latency(trajectories),
        }
        
        # Add mutation-specific context
        if mutation_type == MutationType.REWRITE:
            context.update(self._prepare_rewrite_context(trajectories, **kwargs))
        elif mutation_type == MutationType.INSERT:
            context.update(self._prepare_insert_context(trajectories, **kwargs))
        elif mutation_type == MutationType.DELETE:
            context.update(self._prepare_delete_context(current_prompt, **kwargs))
        elif mutation_type == MutationType.COMPRESS:
            context.update(self._prepare_compress_context(current_prompt, **kwargs))
        
        # Generate reflection prompt
        template = self.REFLECTION_TEMPLATES[mutation_type]
        reflection_prompt = template.render(**context)
        
        # Get mutated prompt from reflection model
        request = InferenceRequest(
            prompt=reflection_prompt,
            max_tokens=2048,
            temperature=0.7
        )
        
        response = await self.reflection_client.generate(request)
        
        # Extract the new prompt from response
        new_prompt = self._extract_new_prompt(response.text, mutation_type)
        
        return new_prompt
    
    def _analyze_trajectories(
        self, 
        trajectories: List[Trajectory], 
        module_id: str
    ) -> List[str]:
        """Analyze trajectories to identify issues."""
        issues = []
        
        if not trajectories:
            return ["No trajectory data available for analysis"]
        
        # Calculate failure rate
        failed_trajectories = [t for t in trajectories if not t.success]
        if failed_trajectories:
            failure_rate = len(failed_trajectories) / len(trajectories)
            if failure_rate > 0.3:
                issues.append(f"High failure rate: {failure_rate:.1%}")
        
        # Analyze latency issues
        latencies = [t.total_latency for t in trajectories]
        avg_latency = sum(latencies) / len(latencies)
        if avg_latency > 10.0:  # More than 10 seconds
            issues.append(f"High average latency: {avg_latency:.1f}s")
        
        # Look for common error patterns
        error_patterns = {}
        for trajectory in failed_trajectories:
            if trajectory.error:
                error_type = trajectory.error.split(':')[0]
                error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
        
        for error_type, count in error_patterns.items():
            if count > 1:
                issues.append(f"Recurring error: {error_type} ({count} times)")
        
        # Module-specific analysis
        module_issues = self._analyze_module_performance(trajectories, module_id)
        issues.extend(module_issues)
        
        return issues[:5]  # Limit to top 5 issues
    
    def _analyze_module_performance(
        self, 
        trajectories: List[Trajectory], 
        module_id: str
    ) -> List[str]:
        """Analyze performance of a specific module."""
        issues = []
        
        module_steps = []
        for trajectory in trajectories:
            for step in trajectory.steps:
                if step.module_id == module_id:
                    module_steps.append(step)
        
        if not module_steps:
            return ["Module not found in trajectories"]
        
        # Check for module-specific errors
        error_steps = [step for step in module_steps if step.error]
        if error_steps:
            error_rate = len(error_steps) / len(module_steps)
            if error_rate > 0.2:
                issues.append(f"Module error rate: {error_rate:.1%}")
        
        # Check for output quality issues (could be extended with more sophisticated analysis)
        empty_outputs = [step for step in module_steps if not step.output_data]
        if empty_outputs:
            empty_rate = len(empty_outputs) / len(module_steps)
            if empty_rate > 0.1:
                issues.append(f"Empty output rate: {empty_rate:.1%}")
        
        return issues
    
    def _calculate_success_rate(self, trajectories: List[Trajectory]) -> float:
        """Calculate success rate from trajectories."""
        if not trajectories:
            return 0.0
        return sum(1 for t in trajectories if t.success) / len(trajectories)
    
    def _calculate_avg_latency(self, trajectories: List[Trajectory]) -> float:
        """Calculate average latency from trajectories."""
        if not trajectories:
            return 0.0
        return sum(t.total_latency for t in trajectories) / len(trajectories)
    
    def _prepare_rewrite_context(self, trajectories: List[Trajectory], **kwargs) -> Dict[str, Any]:
        """Prepare context for rewrite mutation."""
        # Get example inputs/outputs from trajectories
        examples = []
        for trajectory in trajectories[:3]:  # Use first 3 trajectories as examples
            examples.append({
                "input": trajectory.input_data,
                "output": trajectory.output_data,
                "success": trajectory.success
            })
        
        return {
            "examples": examples,
            "input_data": trajectories[0].input_data if trajectories else {},
            "output_data": trajectories[0].output_data if trajectories else {},
            "total_latency": trajectories[0].total_latency if trajectories else 0,
            "success": trajectories[0].success if trajectories else False,
            "error": trajectories[0].error if trajectories else None,
        }
    
    def _prepare_insert_context(self, trajectories: List[Trajectory], **kwargs) -> Dict[str, Any]:
        """Prepare context for insert mutation."""
        # Find successful examples to learn from
        successful_trajectories = [t for t in trajectories if t.success]
        examples = []
        
        for trajectory in successful_trajectories[:3]:
            examples.append({
                "input": trajectory.input_data,
                "expected": trajectory.output_data
            })
        
        return {"examples": examples}
    
    def _prepare_delete_context(self, current_prompt: str, **kwargs) -> Dict[str, Any]:
        """Prepare context for delete mutation."""
        target_reduction = kwargs.get("target_reduction", 20)  # Default 20% reduction
        
        return {
            "prompt_length": len(current_prompt),
            "target_reduction": target_reduction
        }
    
    def _prepare_compress_context(self, current_prompt: str, **kwargs) -> Dict[str, Any]:
        """Prepare context for compress mutation."""
        current_length = len(current_prompt.split())
        compression_ratio = kwargs.get("compression_ratio", 0.7)  # Default 30% reduction
        target_length = int(current_length * compression_ratio)
        
        # Extract key requirements (this could be more sophisticated)
        key_requirements = [
            "Maintain output format",
            "Preserve core instructions",
            "Keep essential examples"
        ]
        
        return {
            "current_length": current_length,
            "target_length": target_length,
            "key_requirements": key_requirements,
            "performance_threshold": kwargs.get("performance_threshold", 90)
        }
    
    def _extract_new_prompt(self, response_text: str, mutation_type: MutationType) -> str:
        """Extract the new prompt from the reflection model's response."""
        # Look for common markers that indicate the start of the new prompt
        markers = [
            "## Improved Prompt",
            "## Enhanced Prompt", 
            "## Simplified Prompt",
            "## Compressed Prompt",
            "New Prompt:",
            "Result:"
        ]
        
        text = response_text.strip()
        
        # Find the marker and extract text after it
        for marker in markers:
            if marker in text:
                parts = text.split(marker, 1)
                if len(parts) > 1:
                    new_prompt = parts[1].strip()
                    # Remove any trailing explanations or metadata
                    lines = new_prompt.split('\n')
                    prompt_lines = []
                    for line in lines:
                        # Stop at explanation markers
                        if line.strip().startswith(('## ', '# ', 'Note:', 'Explanation:')):
                            break
                        prompt_lines.append(line)
                    return '\n'.join(prompt_lines).strip()
        
        # If no marker found, try to extract the last substantial paragraph
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            return paragraphs[-1]
        
        # Fallback: return the whole response
        return text
    
    async def algorithm3_reflective_mutation(
        self,
        system: CompoundAISystem,
        training_dataset: List[Dict[str, Any]],
        inference_client: Any,  # InferenceClient
        evaluator: Any,  # Evaluator
        minibatch_size: int = 3
    ) -> Optional[CompoundAISystem]:
        """
        Algorithm 3: Reflective Prompt Mutation from the GEPA paper.
        
        Implements the exact algorithm:
        1. Select target module (round-robin)
        2. Generate rollouts on minibatch
        3. Extract execution traces and evaluation feedback
        4. Use LLM reflection to analyze and improve prompt
        5. Return new system with updated prompt
        
        Args:
            system: The compound AI system to mutate
            training_dataset: Training data for generating rollouts
            inference_client: Client for system execution
            evaluator: Evaluator for assessing performance
            minibatch_size: Size of minibatch for rollouts
            
        Returns:
            New system with mutated prompt, or None if mutation failed
        """
        
        # Step 1: Select target module (round-robin as mentioned in paper)
        # Use intelligent module selection if available and enabled
        try:
            if (hasattr(self, 'config') and hasattr(self.config, 'advanced') and 
                self.config.advanced.module_selection_strategy == "intelligent"):
                target_module_id = self._select_target_module_intelligent(system, context)
            else:
                target_module_id = self._select_target_module_round_robin(system)
        except (ImportError, AttributeError):
            target_module_id = self._select_target_module_round_robin(system)
        if not target_module_id:
            return None
        
        # Step 2: Sample minibatch from training dataset
        minibatch = random.sample(
            training_dataset, 
            min(minibatch_size, len(training_dataset))
        )
        
        # Step 3: Generate rollouts and collect execution traces
        trajectories = []
        evaluation_traces = []
        
        for data_point in minibatch:
            try:
                # Execute system and capture detailed trajectory
                trajectory = await self._execute_with_detailed_tracing(
                    system, data_point, inference_client
                )
                trajectories.append(trajectory)
                
                # Get evaluation trace with rich diagnostic information
                eval_trace = await self._get_detailed_evaluation_trace(
                    trajectory, data_point, evaluator
                )
                evaluation_traces.append(eval_trace)
                
            except Exception as e:
                # Create error trajectory for reflection
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
        
        # Step 4: Perform LLM-based reflection using evaluation traces
        new_prompt = await self._perform_algorithm3_reflection(
            target_module_id,
            system.modules[target_module_id].prompt,
            trajectories,
            evaluation_traces
        )
        
        if not new_prompt:
            return None
        
        # Step 5: Create new system with updated prompt
        new_system = system.update_module(target_module_id, new_prompt)
        
        return new_system
    
    def _select_target_module_round_robin(self, system: CompoundAISystem) -> Optional[str]:
        """Select target module using round-robin strategy as mentioned in the paper."""
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
    
    def _select_target_module_intelligent(self, system: CompoundAISystem, context: Dict[str, Any]) -> Optional[str]:
        """Select target module using intelligent multi-criteria approach."""
        try:
            from ..algorithms.advanced.intelligent_selection import IntelligentModuleSelector, SelectionContext
            
            # Initialize selector if not exists
            if not hasattr(self, '_intelligent_selector'):
                self._intelligent_selector = IntelligentModuleSelector()
            
            # Create selection context
            performance_history = getattr(self, 'performance_history', {})
            mutation_history = getattr(self, 'mutation_history', {})
            
            selection_context = SelectionContext(
                performance_history=performance_history,
                mutation_history=mutation_history,
                current_generation=context.get('generation', 0),
                budget_remaining=context.get('budget_remaining', 1.0)
            )
            
            # Select module using intelligent approach
            selected_module = self._intelligent_selector.select_target_module(system, selection_context)
            
            # Update round-robin counter for compatibility
            if hasattr(self, 'module_selection_counter'):
                self.module_selection_counter[selected_module] = self.module_selection_counter.get(selected_module, 0) + 1
            
            return selected_module
            
        except ImportError:
            # Fallback to round-robin if advanced algorithms not available
            return self._select_target_module_round_robin(system)
    
    async def _execute_with_detailed_tracing(
        self,
        system: CompoundAISystem,
        data_point: Dict[str, Any],
        inference_client: Any
    ) -> Trajectory:
        """Execute system with detailed tracing for each module step."""
        import time
        
        start_time = time.time()
        steps = []
        success = False
        error = None
        output_data = {}
        
        try:
            # Execute system and capture intermediate steps
            # This is a simplified version - in full implementation,
            # each module execution would be individually traced
            output_data = await system.execute(data_point, inference_client)
            
            # Create detailed steps for each module
            for module_id, module in system.modules.items():
                step = TrajectoryStep(
                    module_id=module_id,
                    input_data=data_point,
                    output_data=output_data,  # In real implementation, this would be module-specific
                    prompt_used=module.prompt,
                    latency=0.1,  # Would be actual module latency
                    error=None
                )
                steps.append(step)
            
            success = True
            
        except Exception as e:
            error = str(e)
            # Create error step for the failed module
            # In real implementation, we'd know which module failed
            for module_id in system.modules.keys():
                step = TrajectoryStep(
                    module_id=module_id,
                    input_data=data_point,
                    output_data={},
                    prompt_used=system.modules[module_id].prompt,
                    latency=0.0,
                    error=error
                )
                steps.append(step)
                break  # Only add error step for first module
        
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
    
    async def _get_detailed_evaluation_trace(
        self,
        trajectory: Trajectory,
        data_point: Dict[str, Any],
        evaluator: Any
    ) -> Dict[str, Any]:
        """Get rich evaluation trace with diagnostic information as mentioned in the paper."""
        try:
            # Get evaluation result with detailed feedback
            result = await evaluator.evaluate_single(
                trajectory.output_data,
                {"expected": data_point.get("expected", {})}
            )
            
            # Create rich diagnostic trace
            trace = {
                "success": trajectory.success,
                "scores": result.scores,
                "execution_steps": len(trajectory.steps),
                "total_latency": trajectory.total_latency,
                "input": trajectory.input_data,
                "output": trajectory.output_data,
                "expected": data_point.get("expected", {})
            }
            
            # Add step-by-step analysis
            if trajectory.steps:
                trace["step_analysis"] = []
                for step in trajectory.steps:
                    step_info = {
                        "module_id": step.module_id,
                        "latency": step.latency,
                        "error": step.error,
                        "has_output": bool(step.output_data)
                    }
                    trace["step_analysis"].append(step_info)
            
            return trace
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_steps": len(trajectory.steps),
                "total_latency": trajectory.total_latency,
                "input": trajectory.input_data,
                "output": trajectory.output_data
            }
    
    async def _perform_algorithm3_reflection(
        self,
        target_module_id: str,
        current_prompt: str,
        trajectories: List[Trajectory],
        evaluation_traces: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Perform LLM-based reflection using evaluation traces as diagnostic signal."""
        
        # Analyze trajectories and evaluation traces
        analysis = self._analyze_trajectories_and_traces(trajectories, evaluation_traces)
        
        # Create reflection prompt using evaluation traces as diagnostic signal
        reflection_prompt = self._create_algorithm3_reflection_prompt(
            target_module_id,
            current_prompt,
            analysis
        )
        
        # Generate reflection using LLM
        request = InferenceRequest(
            prompt=reflection_prompt,
            max_tokens=2048,
            temperature=0.7
        )
        
        try:
            response = await self.reflection_client.generate(request)
            
            # Extract new prompt from reflection
            new_prompt = self._extract_new_prompt(response.text, MutationType.REWRITE)
            
            return new_prompt
            
        except Exception as e:
            print(f"Algorithm 3 reflection failed: {e}")
            return None
    
    def _analyze_trajectories_and_traces(
        self,
        trajectories: List[Trajectory],
        evaluation_traces: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze trajectories and evaluation traces for reflection."""
        
        successful_trajectories = [t for t in trajectories if t.success]
        failed_trajectories = [t for t in trajectories if not t.success]
        
        successful_traces = [t for t in evaluation_traces if t.get("success", False)]
        failed_traces = [t for t in evaluation_traces if not t.get("success", False)]
        
        analysis = {
            "success_rate": len(successful_trajectories) / max(len(trajectories), 1),
            "avg_latency": sum(t.total_latency for t in trajectories) / max(len(trajectories), 1),
            "successful_examples": [],
            "failed_examples": [],
            "common_errors": [],
            "evaluation_feedback": []
        }
        
        # Collect successful examples
        for traj, trace in zip(successful_trajectories, successful_traces):
            example = {
                "input": traj.input_data,
                "output": traj.output_data,
                "scores": trace.get("scores", {}),
                "latency": traj.total_latency
            }
            analysis["successful_examples"].append(example)
        
        # Collect failure examples with diagnostic info
        for traj, trace in zip(failed_trajectories, failed_traces):
            example = {
                "input": traj.input_data,
                "output": traj.output_data,
                "error": traj.error,
                "scores": trace.get("scores", {}),
                "expected": trace.get("expected", {})
            }
            analysis["failed_examples"].append(example)
        
        # Extract common error patterns
        error_counts = {}
        for traj in failed_trajectories:
            if traj.error:
                error_type = traj.error.split(':')[0].strip()
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        analysis["common_errors"] = [
            f"{error_type} ({count}x)" 
            for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # Collect evaluation feedback
        for trace in evaluation_traces:
            if "scores" in trace:
                analysis["evaluation_feedback"].append(trace["scores"])
        
        return analysis
    
    def _create_algorithm3_reflection_prompt(
        self,
        module_id: str,
        current_prompt: str,
        analysis: Dict[str, Any]
    ) -> str:
        """Create reflection prompt using evaluation traces as diagnostic signal."""
        
        # Format successful examples
        success_examples = ""
        for i, example in enumerate(analysis["successful_examples"][:2]):
            success_examples += f"""
**Success Example {i+1}:**
- Input: {example['input']}
- Output: {example['output']}
- Scores: {example['scores']}
- Latency: {example['latency']:.2f}s
"""
        
        # Format failure examples
        failure_examples = ""
        for i, example in enumerate(analysis["failed_examples"][:2]):
            failure_examples += f"""
**Failure Example {i+1}:**
- Input: {example['input']}
- Output: {example['output']}
- Expected: {example['expected']}
- Error: {example.get('error', 'Unknown error')}
- Scores: {example['scores']}
"""
        
        # Create reflection prompt based on paper's approach
        reflection_prompt = f"""
# GEPA Algorithm 3: Reflective Prompt Mutation

## Current Module Analysis
**Module ID**: {module_id}
**Current Prompt**:
```
{current_prompt}
```

## Execution Trace Analysis

### Performance Metrics
- Success Rate: {analysis['success_rate']:.1%}
- Average Latency: {analysis['avg_latency']:.2f}s
- Common Errors: {', '.join(analysis['common_errors'][:3]) if analysis['common_errors'] else 'None'}

### Successful Executions
{success_examples if success_examples else 'No successful executions found'}

### Failed Executions
{failure_examples if failure_examples else 'No failed executions found'}

### Evaluation Feedback
{analysis['evaluation_feedback'][:3] if analysis['evaluation_feedback'] else 'No evaluation feedback available'}

## Reflection Task

Based on the execution traces and evaluation feedback above, analyze the current prompt for module "{module_id}" and identify specific issues that led to failures or suboptimal performance.

Use the evaluation traces as diagnostic signals to understand:
1. What patterns lead to successful vs failed executions?
2. What specific aspects of the prompt need improvement?
3. How can the prompt better handle the input patterns that currently fail?
4. What constraints or examples should be added?

## Improved Prompt

Provide a revised prompt that addresses the identified issues based on the diagnostic information:

```
[Your improved prompt here]
```

## Key Changes Made
Explain the specific changes and how they address the diagnostic findings:
"""
        
        return reflection_prompt