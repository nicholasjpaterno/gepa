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
    """Handles reflective mutation of prompts based on trajectory analysis."""
    
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
    
    def __init__(self, reflection_client: InferenceClient):
        self.reflection_client = reflection_client
    
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