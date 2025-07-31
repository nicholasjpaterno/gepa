"""Prompt Structure Evolution for MetaOrchestrator."""

import logging
import re
import random
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from .config import PromptConfig

logger = logging.getLogger(__name__)


@dataclass
class LinguisticFeature:
    """Represents a linguistic feature in prompts."""
    feature_type: str  # "word", "phrase", "pattern", "structure"
    content: str
    frequency: int = 0
    success_correlation: float = 0.0
    context: str = ""


@dataclass
class PromptComponent:
    """Individual component of a prompt structure."""
    component_type: str  # "instruction", "example", "constraint", "format"
    content: str
    position: int
    effectiveness_score: float = 0.0
    interactions: Dict[str, float] = None
    
    def __post_init__(self):
        if self.interactions is None:
            self.interactions = {}


@dataclass 
class PromptStructure:
    """Complete prompt structure representation."""
    components: List[PromptComponent]
    grammar_rules: Dict[str, Any]
    semantic_patterns: List[str]
    effectiveness_score: float = 0.0
    
    def to_prompt(self, context: Dict[str, Any] = None) -> str:
        """Convert structure to actual prompt text."""
        sorted_components = sorted(self.components, key=lambda x: x.position)
        prompt_parts = [comp.content for comp in sorted_components]
        
        # Apply context substitution if provided
        prompt_text = "\n\n".join(prompt_parts)
        if context:
            for key, value in context.items():
                prompt_text = prompt_text.replace(f"{{{key}}}", str(value))
        
        return prompt_text


class PromptGrammarEvolver:
    """Evolves prompt grammar and syntax patterns."""
    
    def __init__(self):
        # Grammar rule templates
        self.grammar_templates = {
            "instruction_pattern": [
                "Please {verb} the following {object}:",
                "Your task is to {verb} {object}:",
                "{verb} the {object} provided below:",
                "I need you to {verb} this {object}:"
            ],
            "constraint_pattern": [
                "Make sure to {constraint}",
                "Remember that {constraint}",
                "It's important to {constraint}",
                "Always {constraint}"
            ],
            "example_pattern": [
                "For example: {example}",
                "Here's an example: {example}",
                "Consider this example: {example}",
                "Example: {example}"
            ]
        }
        
        # Syntax patterns for different task types
        self.task_patterns = {
            "code_generation": {
                "opening": ["Write a function that", "Create a method to", "Implement"],
                "requirements": ["The function should", "Make sure it", "Ensure that"],
                "output_format": ["Return", "Output", "The result should be"]
            },
            "summarization": {
                "opening": ["Summarize", "Create a summary of", "Provide a concise overview"],
                "requirements": ["Focus on", "Include", "Highlight"],
                "output_format": ["The summary should", "Format the output", "Present as"]
            },
            "qa": {
                "opening": ["Answer the question", "Provide an answer", "Respond to"],
                "requirements": ["Base your answer on", "Consider", "Make sure to"],
                "output_format": ["Answer:", "Response:", "The answer is"]
            }
        }
    
    def mutate_syntax(self, current_prompts: Dict[str, str]) -> Dict[str, str]:
        """Mutate syntax patterns in current prompts."""
        mutated_prompts = {}
        
        for module_id, prompt in current_prompts.items():
            mutated_prompt = self._apply_syntax_mutations(prompt)
            mutated_prompts[module_id] = mutated_prompt
        
        return mutated_prompts
    
    def _apply_syntax_mutations(self, prompt: str) -> str:
        """Apply various syntax mutations to a prompt."""
        mutations = [
            self._mutate_instruction_style,
            self._mutate_constraint_phrasing,
            self._mutate_example_format,
            self._mutate_output_specification
        ]
        
        # Apply 1-2 random mutations
        selected_mutations = random.sample(mutations, k=min(2, len(mutations)))
        
        mutated_prompt = prompt
        for mutation_func in selected_mutations:
            mutated_prompt = mutation_func(mutated_prompt)
        
        return mutated_prompt
    
    def _mutate_instruction_style(self, prompt: str) -> str:
        """Change instruction phrasing style."""
        # Simple pattern replacements
        patterns = [
            (r"Please (.+):", r"Your task is to \1:"),
            (r"I need you to (.+)", r"Please \1"),
            (r"You should (.+)", r"Make sure to \1"),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                prompt = re.sub(pattern, replacement, prompt, count=1, flags=re.IGNORECASE)
                break
        
        return prompt
    
    def _mutate_constraint_phrasing(self, prompt: str) -> str:
        """Change constraint phrasing."""
        constraint_starters = ["Make sure", "Remember", "It's important", "Always", "Ensure that"]
        
        # Find existing constraints and rephrase
        lines = prompt.split('\n')
        for i, line in enumerate(lines):
            for starter in constraint_starters:
                if line.strip().startswith(starter):
                    new_starter = random.choice([s for s in constraint_starters if s != starter])
                    lines[i] = line.replace(starter, new_starter, 1)
                    break
        
        return '\n'.join(lines)
    
    def _mutate_example_format(self, prompt: str) -> str:
        """Change example formatting."""
        example_patterns = [
            (r"For example:", "Here's an example:"),
            (r"Example:", "Consider this example:"),
            (r"Here's an example:", "For instance:"),
        ]
        
        for old_pattern, new_pattern in example_patterns:
            if old_pattern in prompt:
                prompt = prompt.replace(old_pattern, new_pattern, 1)
                break
        
        return prompt
    
    def _mutate_output_specification(self, prompt: str) -> str:
        """Change output format specification."""
        output_patterns = [
            (r"Return (.+)", r"Output \1"),
            (r"The result should be (.+)", r"Format the output as \1"),
            (r"Provide (.+)", r"Generate \1"),
        ]
        
        for pattern, replacement in output_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                prompt = re.sub(pattern, replacement, prompt, count=1, flags=re.IGNORECASE)
                break
        
        return prompt


class SemanticPatternDiscoverer:
    """Discovers semantic patterns in successful prompts."""
    
    def __init__(self):
        self.success_patterns = []
        self.failure_patterns = []
        self.pattern_effectiveness = {}
    
    def discover_new_patterns(self, performance_feedback: List[float]) -> Dict[str, List[str]]:
        """Discover new semantic patterns based on performance."""
        if len(performance_feedback) < 5:
            return {"patterns": []}
        
        # Analyze performance distribution
        high_performance = [i for i, score in enumerate(performance_feedback) if score > 0.7]
        low_performance = [i for i, score in enumerate(performance_feedback) if score < 0.3]
        
        discovered_patterns = {
            "high_performance_patterns": self._extract_success_patterns(high_performance),
            "low_performance_patterns": self._extract_failure_patterns(low_performance),
            "emerging_patterns": self._identify_emerging_patterns(performance_feedback)
        }
        
        return discovered_patterns
    
    def _extract_success_patterns(self, high_performance_indices: List[int]) -> List[str]:
        """Extract patterns from high-performing prompts."""
        success_patterns = [
            "specific_instructions",
            "clear_examples",
            "explicit_constraints",
            "structured_output",
            "step_by_step_guidance"
        ]
        
        # In practice, would analyze actual prompt text
        return random.sample(success_patterns, k=min(3, len(success_patterns)))
    
    def _extract_failure_patterns(self, low_performance_indices: List[int]) -> List[str]:
        """Extract patterns from low-performing prompts."""
        failure_patterns = [
            "vague_instructions",
            "ambiguous_requirements",
            "missing_examples",
            "unclear_output_format",
            "conflicting_constraints"
        ]
        
        return random.sample(failure_patterns, k=min(2, len(failure_patterns)))
    
    def _identify_emerging_patterns(self, performance_feedback: List[float]) -> List[str]:
        """Identify emerging successful patterns."""
        # Analyze performance trends
        if len(performance_feedback) < 10:
            return []
        
        recent_performance = performance_feedback[-5:]
        avg_recent = np.mean(recent_performance)
        
        if avg_recent > np.mean(performance_feedback):
            return ["positive_trend_pattern", "effective_recent_changes"]
        
        return []


class CompositionalPromptCompositor:
    """Composes hybrid prompts from successful components."""
    
    def __init__(self):
        self.component_library = {
            "instructions": [
                "Analyze the following {input_type} carefully",
                "Process the {input_type} according to these requirements",
                "Examine the provided {input_type} and"
            ],
            "requirements": [
                "Focus on accuracy and completeness",
                "Maintain clarity and coherence",
                "Ensure the output is well-structured"
            ],
            "examples": [
                "Example: {example_content}",
                "For instance: {example_content}",
                "Consider this case: {example_content}"
            ],
            "output_specs": [
                "Format your response as: {format}",
                "Structure the output using: {format}",
                "Present the result in: {format}"
            ]
        }
    
    def compose_hybrid_prompts(self, current_prompts: Dict[str, str]) -> Dict[str, str]:
        """Compose new hybrid prompts from existing successful components."""
        hybrid_prompts = {}
        
        for module_id, prompt in current_prompts.items():
            # Extract components from current prompt
            components = self._extract_components(prompt)
            
            # Combine with library components
            hybrid_components = self._create_hybrid_components(components)
            
            # Compose new prompt
            hybrid_prompt = self._compose_from_components(hybrid_components)
            hybrid_prompts[module_id] = hybrid_prompt
        
        return hybrid_prompts
    
    def _extract_components(self, prompt: str) -> List[PromptComponent]:
        """Extract reusable components from a prompt."""
        components = []
        lines = prompt.split('\n')
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            
            # Classify line type
            component_type = self._classify_line_type(line)
            
            component = PromptComponent(
                component_type=component_type,
                content=line.strip(),
                position=i
            )
            components.append(component)
        
        return components
    
    def _classify_line_type(self, line: str) -> str:
        """Classify the type of a prompt line."""
        line_lower = line.lower().strip()
        
        if any(word in line_lower for word in ["analyze", "process", "examine", "please"]):
            return "instruction"
        elif any(word in line_lower for word in ["example", "for instance", "consider"]):
            return "example"
        elif any(word in line_lower for word in ["format", "output", "result", "response"]):
            return "output_spec"
        elif any(word in line_lower for word in ["make sure", "ensure", "remember", "important"]):
            return "constraint"
        else:
            return "general"
    
    def _create_hybrid_components(self, existing_components: List[PromptComponent]) -> List[PromptComponent]:
        """Create hybrid components by mixing existing and library components."""
        hybrid_components = existing_components.copy()
        
        # Add complementary components from library
        existing_types = {comp.component_type for comp in existing_components}
        
        for comp_type, templates in self.component_library.items():
            if comp_type not in existing_types and random.random() < 0.3:
                # Add component from library
                template = random.choice(templates)
                
                new_component = PromptComponent(
                    component_type=comp_type,
                    content=template,
                    position=len(hybrid_components)
                )
                hybrid_components.append(new_component)
        
        return hybrid_components
    
    def _compose_from_components(self, components: List[PromptComponent]) -> str:
        """Compose final prompt from components."""
        # Sort by logical order
        type_order = {"instruction": 0, "constraint": 1, "example": 2, "output_spec": 3, "general": 4}
        
        sorted_components = sorted(
            components,
            key=lambda x: (type_order.get(x.component_type, 99), x.position)
        )
        
        prompt_lines = [comp.content for comp in sorted_components]
        return '\n\n'.join(prompt_lines)


class PromptStructureEvolver:
    """
    Evolves prompt templates and communication patterns.
    
    Treats prompt structure as an evolvable genome rather than fixed text,
    using compositional evolution to discover optimal prompt architectures.
    """
    
    def __init__(self, config: PromptConfig):
        self.config = config
        
        # Prompt grammar and syntax evolution
        self.grammar_evolver = PromptGrammarEvolver()
        
        # Semantic pattern discovery
        self.pattern_discoverer = SemanticPatternDiscoverer()
        
        # Compositional prompt generation
        self.compositor = CompositionalPromptCompositor()
        
        # Component analysis
        self.component_effectiveness = defaultdict(list)
        self.successful_patterns = []
        self.prompt_evolution_history = []
        
        logger.info("PromptStructureEvolver initialized with compositional evolution")
    
    async def evolve_prompt_structure(
        self,
        current_prompts: Dict[str, str],
        performance_feedback: List[float]
    ) -> Optional[Dict[str, Any]]:
        """
        Evolve prompt structure based on performance patterns.
        """
        if len(performance_feedback) < 3:
            logger.debug("Insufficient feedback for prompt structure evolution")
            return None
        
        logger.info(f"Evolving prompt structure based on {len(performance_feedback)} feedback points")
        
        # Analyze which prompt components correlate with success
        component_analysis = self.analyze_prompt_components(current_prompts, performance_feedback)
        
        # Generate new structural variations
        structural_mutations = []
        
        if self.config.grammar_evolution_enabled:
            grammar_mutation = self.grammar_evolver.mutate_syntax(current_prompts)
            structural_mutations.append(("grammar", grammar_mutation))
        
        if self.config.semantic_pattern_discovery:
            pattern_mutation = self.pattern_discoverer.discover_new_patterns(performance_feedback)
            structural_mutations.append(("patterns", pattern_mutation))
        
        if self.config.compositional_generation:
            compositional_mutation = self.compositor.compose_hybrid_prompts(current_prompts)
            structural_mutations.append(("compositional", compositional_mutation))
        
        # Evaluate structural fitness
        candidate_structures = []
        for mutation_type, mutation in structural_mutations:
            fitness = self.evaluate_structure_fitness(mutation, component_analysis)
            candidate_structures.append((mutation_type, mutation, fitness))
        
        if not candidate_structures:
            return None
        
        # Select best structure
        best_type, best_structure, best_fitness = max(
            candidate_structures, 
            key=lambda x: x[2]
        )
        
        logger.info(f"Selected best structure: {best_type} (fitness: {best_fitness:.3f})")
        
        # Track evolution
        self.prompt_evolution_history.append({
            "type": best_type,
            "structure": best_structure,
            "fitness": best_fitness,
            "component_analysis": component_analysis
        })
        
        return {
            "evolution_type": best_type,
            "updated_prompts": best_structure if isinstance(best_structure, dict) else current_prompts,
            "fitness_score": best_fitness,
            "component_insights": component_analysis
        }
    
    def analyze_prompt_components(
        self,
        prompts: Dict[str, str],
        feedback: List[float]
    ) -> Dict[str, Any]:
        """
        Identify which prompt components contribute to success.
        """
        if len(feedback) != len(prompts) and len(feedback) > 0:
            # Use average feedback for all prompts
            avg_feedback = np.mean(feedback)
            feedback_per_prompt = {module_id: avg_feedback for module_id in prompts.keys()}
        else:
            # Map feedback to prompts (simplified)
            prompt_ids = list(prompts.keys())
            feedback_per_prompt = {
                prompt_ids[i % len(prompt_ids)]: feedback[i] 
                for i in range(len(feedback))
            }
        
        # Extract linguistic features from successful vs unsuccessful prompts
        successful_prompts = []
        unsuccessful_prompts = []
        
        for module_id, prompt in prompts.items():
            avg_score = feedback_per_prompt.get(module_id, 0.5)
            if avg_score > 0.6:
                successful_prompts.append(prompt)
            elif avg_score < 0.4:
                unsuccessful_prompts.append(prompt)
        
        if not successful_prompts:
            # If no clearly successful prompts, use all
            successful_prompts = list(prompts.values())
        
        # Extract features
        successful_features = self.extract_linguistic_features(successful_prompts)
        unsuccessful_features = self.extract_linguistic_features(unsuccessful_prompts)
        
        # Identify discriminative patterns
        discriminative_patterns = self.find_discriminative_patterns(
            successful_features, unsuccessful_features
        )
        
        return {
            "successful_features": successful_features,
            "unsuccessful_features": unsuccessful_features,
            "discriminative_patterns": discriminative_patterns,
            "success_rate": len(successful_prompts) / len(prompts) if prompts else 0,
            "total_prompts_analyzed": len(prompts)
        }
    
    def extract_linguistic_features(self, prompts: List[str]) -> List[LinguisticFeature]:
        """Extract linguistic features from prompts."""
        features = []
        
        for prompt in prompts:
            # Word-level features
            words = prompt.lower().split()
            word_counts = defaultdict(int)
            for word in words:
                word_counts[word] += 1
            
            for word, count in word_counts.items():
                if count > 1:  # Only significant words
                    features.append(LinguisticFeature(
                        feature_type="word",
                        content=word,
                        frequency=count
                    ))
            
            # Phrase-level features
            sentences = re.split(r'[.!?]', prompt)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:  # Meaningful sentences
                    features.append(LinguisticFeature(
                        feature_type="phrase",
                        content=sentence[:50],  # Truncate for analysis
                        frequency=1
                    ))
            
            # Pattern features
            patterns = [
                r'please .+',
                r'make sure .+',
                r'for example.+',
                r'the result should.+',
                r'format .+ as.+'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, prompt.lower())
                for match in matches:
                    features.append(LinguisticFeature(
                        feature_type="pattern",
                        content=match[:30],
                        frequency=1
                    ))
        
        return features
    
    def find_discriminative_patterns(
        self,
        successful_features: List[LinguisticFeature],
        unsuccessful_features: List[LinguisticFeature]
    ) -> List[Dict[str, Any]]:
        """Identify patterns that discriminate between success and failure."""
        
        # Count feature occurrences
        success_counts = defaultdict(int)
        failure_counts = defaultdict(int)
        
        for feature in successful_features:
            key = f"{feature.feature_type}:{feature.content}"
            success_counts[key] += feature.frequency
        
        for feature in unsuccessful_features:
            key = f"{feature.feature_type}:{feature.content}"
            failure_counts[key] += feature.frequency
        
        # Find discriminative patterns
        discriminative_patterns = []
        
        for feature_key in success_counts:
            success_count = success_counts[feature_key]
            failure_count = failure_counts.get(feature_key, 0)
            
            # Calculate discrimination score
            total_count = success_count + failure_count
            if total_count > 0:
                success_rate = success_count / total_count
                
                # Strong positive discriminator
                if success_rate > 0.7 and success_count >= 2:
                    discriminative_patterns.append({
                        "pattern": feature_key,
                        "success_rate": success_rate,
                        "success_count": success_count,
                        "failure_count": failure_count,
                        "discrimination_type": "positive"
                    })
        
        # Also find negative discriminators (patterns associated with failure)
        for feature_key in failure_counts:
            success_count = success_counts.get(feature_key, 0)
            failure_count = failure_counts[feature_key]
            
            total_count = success_count + failure_count
            if total_count > 0:
                failure_rate = failure_count / total_count
                
                if failure_rate > 0.7 and failure_count >= 2:
                    discriminative_patterns.append({
                        "pattern": feature_key,
                        "success_rate": 1 - failure_rate,
                        "success_count": success_count,
                        "failure_count": failure_count,
                        "discrimination_type": "negative"
                    })
        
        # Sort by discrimination strength
        discriminative_patterns.sort(
            key=lambda x: abs(x["success_rate"] - 0.5) * (x["success_count"] + x["failure_count"]),
            reverse=True
        )
        
        return discriminative_patterns[:10]  # Top 10 patterns
    
    def evaluate_structure_fitness(
        self,
        structure: Any,
        component_analysis: Dict[str, Any]
    ) -> float:
        """Evaluate fitness of a prompt structure."""
        
        if not isinstance(structure, dict):
            return 0.0
        
        fitness_score = 0.0
        
        for module_id, prompt in structure.items():
            if not isinstance(prompt, str):
                continue
            
            # Base fitness from discriminative patterns
            for pattern_info in component_analysis.get("discriminative_patterns", []):
                pattern = pattern_info["pattern"].split(":", 1)[-1]  # Remove type prefix
                
                if pattern.lower() in prompt.lower():
                    if pattern_info["discrimination_type"] == "positive":
                        fitness_score += pattern_info["success_rate"] * 0.1
                    else:
                        fitness_score -= (1 - pattern_info["success_rate"]) * 0.1
            
            # Structure quality metrics
            prompt_length = len(prompt.split())
            if 20 <= prompt_length <= 150:  # Optimal length range
                fitness_score += 0.1
            
            # Clarity indicators
            if any(word in prompt.lower() for word in ["please", "make sure", "example", "format"]):
                fitness_score += 0.05
            
            # Avoid negative patterns
            if any(word in prompt.lower() for word in ["vague", "unclear", "confusing"]):
                fitness_score -= 0.1
        
        # Normalize fitness score
        return max(0.0, min(1.0, fitness_score))
    
    def update_component_analyzer(
        self,
        prompt_changes: Dict[str, Any],
        performance_feedback: float
    ) -> None:
        """Update component analysis with new prompt performance data."""
        
        # Track component effectiveness
        for change_type, changes in prompt_changes.items():
            self.component_effectiveness[change_type].append(performance_feedback)
        
        # Update successful patterns
        if performance_feedback > 0.7:
            self.successful_patterns.append({
                "changes": prompt_changes,
                "performance": performance_feedback,
                "timestamp": "current"  # In practice, would use actual timestamp
            })
        
        logger.debug(f"Updated component analyzer: {change_type} -> {performance_feedback:.3f}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get prompt evolution metrics."""
        
        # Component effectiveness statistics
        component_stats = {}
        for comp_type, performances in self.component_effectiveness.items():
            if performances:
                component_stats[comp_type] = {
                    "mean_performance": np.mean(performances),
                    "max_performance": np.max(performances),
                    "min_performance": np.min(performances),
                    "count": len(performances)
                }
        
        return {
            "evolution_rounds": len(self.prompt_evolution_history),
            "successful_patterns": len(self.successful_patterns),
            "component_effectiveness": component_stats,
            "recent_evolution_types": [
                h["type"] for h in self.prompt_evolution_history[-5:]
            ] if self.prompt_evolution_history else [],
            "average_fitness": np.mean([
                h["fitness"] for h in self.prompt_evolution_history
            ]) if self.prompt_evolution_history else 0.0
        }