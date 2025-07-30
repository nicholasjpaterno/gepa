"""Sophisticated compatibility analysis replacing simple same-system heuristics."""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import re
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)


class CompatibilityDimension(Enum):
    """Different dimensions of compatibility analysis."""
    SEMANTIC = "semantic"
    STYLE = "style"
    INPUT_OUTPUT = "input_output"
    PERFORMANCE = "performance"
    LOGICAL_FLOW = "logical_flow"


@dataclass
class CompatibilityAnalysis:
    """Comprehensive compatibility analysis result."""
    semantic_compatibility: float
    style_consistency: float
    io_compatibility: float
    performance_correlation: float
    logical_flow: float
    overall_score: float
    explanation: str
    confidence: float


@dataclass
class InteractionContext:
    """Context for module interaction analysis."""
    workflow_position: Dict[str, int]
    data_flow: Dict[str, List[str]]
    execution_history: Dict[str, List[Dict[str, Any]]]
    performance_metrics: Dict[str, Dict[str, float]]


class SemanticSimilarityAnalyzer:
    """Analyze semantic similarity between prompts using multiple approaches."""
    
    def __init__(self):
        self.concept_cache: Dict[str, List[str]] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
    def calculate_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate semantic similarity using multiple approaches."""
        
        try:
            # Approach 1: Embedding-based similarity (simplified)
            embedding_sim = self._embedding_similarity(prompt1, prompt2)
            
            # Approach 2: Concept overlap analysis
            concept_sim = self._concept_overlap_similarity(prompt1, prompt2)
            
            # Approach 3: Instruction pattern matching
            pattern_sim = self._instruction_pattern_similarity(prompt1, prompt2)
            
            # Approach 4: Lexical similarity
            lexical_sim = self._lexical_similarity(prompt1, prompt2)
            
            # Weighted combination
            similarity = (
                0.3 * embedding_sim +
                0.3 * concept_sim +
                0.2 * pattern_sim +
                0.2 * lexical_sim
            )
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.debug(f"Semantic similarity calculation failed: {e}")
            return 0.5  # Default moderate similarity
    
    def _embedding_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate similarity using word embeddings (simplified implementation)."""
        
        # Simplified implementation - in production would use sentence transformers
        # or other pre-trained embedding models
        
        # Extract key terms
        terms1 = self._extract_key_terms(prompt1)
        terms2 = self._extract_key_terms(prompt2)
        
        if not terms1 or not terms2:
            return 0.5
        
        # Simplified Jaccard similarity for key terms
        intersection = len(set(terms1) & set(terms2))
        union = len(set(terms1) | set(terms2))
        
        if union == 0:
            return 1.0
        
        return intersection / union
    
    def _concept_overlap_similarity(self, prompt1: str, prompt2: str) -> float:
        """Analyze concept overlap between prompts."""
        
        concepts1 = self._extract_concepts(prompt1)
        concepts2 = self._extract_concepts(prompt2)
        
        if not concepts1 or not concepts2:
            return 0.3
        
        # Calculate concept overlap
        common_concepts = len(set(concepts1) & set(concepts2))
        total_concepts = len(set(concepts1) | set(concepts2))
        
        if total_concepts == 0:
            return 1.0
        
        overlap_ratio = common_concepts / total_concepts
        
        # Boost similarity if core concepts match
        core_concepts = ['analyze', 'generate', 'classify', 'summarize', 'extract', 'transform']
        core_matches = sum(1 for concept in core_concepts 
                          if concept in concepts1 and concept in concepts2)
        
        core_boost = min(0.3, core_matches * 0.1)
        
        return min(1.0, overlap_ratio + core_boost)
    
    def _instruction_pattern_similarity(self, prompt1: str, prompt2: str) -> float:
        """Analyze similarity in instruction patterns."""
        
        patterns1 = self._extract_instruction_patterns(prompt1)
        patterns2 = self._extract_instruction_patterns(prompt2)
        
        if not patterns1 or not patterns2:
            return 0.4
        
        # Compare patterns
        pattern_matches = 0
        total_patterns = len(set(patterns1 + patterns2))
        
        for pattern in patterns1:
            if pattern in patterns2:
                pattern_matches += 1
        
        if total_patterns == 0:
            return 1.0
        
        return pattern_matches / len(patterns1) if patterns1 else 0.0
    
    def _lexical_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate lexical similarity between prompts."""
        
        # Tokenize and normalize
        tokens1 = set(self._tokenize(prompt1.lower()))
        tokens2 = set(self._tokenize(prompt2.lower()))
        
        if not tokens1 or not tokens2:
            return 0.5
        
        # Jaccard similarity
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_key_terms(self, prompt: str) -> List[str]:
        """Extract key terms from prompt."""
        
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'you', 'your', 'i', 'my', 'me', 'we', 'our', 'us'}
        
        words = self._tokenize(prompt.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_terms
    
    def _extract_concepts(self, prompt: str) -> List[str]:
        """Extract semantic concepts from prompt."""
        
        cache_key = prompt[:100]  # Use first 100 chars as cache key
        if cache_key in self.concept_cache:
            return self.concept_cache[cache_key]
        
        concepts = []
        
        # Action verbs (indicating what the module does)
        action_verbs = re.findall(r'\b(?:analyze|generate|create|classify|summarize|extract|transform|process|convert|translate|identify|detect|compare|evaluate|assess|review|check|validate|format|structure|organize|rank|score|measure|calculate|compute|determine|find|search|locate|retrieve|filter|select|choose|pick|sort|group|cluster|categorize|label|tag|annotate|mark|highlight|emphasize|focus|prioritize|optimize|improve|enhance|refine|polish|edit|revise|update|modify|change|adjust|adapt|customize|personalize|tailor|fit|match|align|sync|coordinate|integrate|combine|merge|join|connect|link|relate|associate|bind|attach|append|add|insert|include|incorporate|embed|inject|apply|use|utilize|employ|leverage|exploit|harness|deploy|implement|execute|run|perform|carry|conduct|operate|manage|handle|control|direct|guide|lead|drive|steer|navigate|route|path|way|method|approach|technique|strategy|plan|design|construct|build|develop|craft|engineer|\w+ly|\w+ing|\w+ed)\b', prompt.lower())
        
        concepts.extend(action_verbs)
        
        # Domain concepts (nouns that indicate domain/subject matter)
        domain_nouns = re.findall(r'\b(?:text|document|article|content|data|information|knowledge|facts|details|specifications|requirements|instructions|guidelines|rules|criteria|standards|metrics|measures|parameters|attributes|properties|features|characteristics|elements|components|parts|sections|segments|chunks|pieces|fragments|tokens|words|sentences|paragraphs|sections|chapters|pages|files|records|entries|items|objects|entities|instances|examples|samples|cases|scenarios|situations|contexts|environments|settings|configurations|options|choices|alternatives|variations|versions|formats|types|kinds|categories|classes|groups|sets|collections|lists|arrays|sequences|series|chains|flows|processes|procedures|steps|stages|phases|tasks|jobs|activities|operations|functions|methods|algorithms|models|systems|frameworks|structures|architectures|designs|patterns|templates|schemas|formats|layouts|arrangements|organizations|hierarchies|taxonomies|classifications|ontologies)\b', prompt.lower())
        
        concepts.extend(domain_nouns)
        
        # Cache results
        self.concept_cache[cache_key] = concepts
        
        return concepts
    
    def _extract_instruction_patterns(self, prompt: str) -> List[str]:
        """Extract instruction patterns from prompt."""
        
        patterns = []
        
        # Imperative patterns
        imperatives = re.findall(r'\b(?:please\s+)?(?:do not|don\'t|never|always|ensure|make sure|be sure|remember to|try to|attempt to|focus on|pay attention to|consider|think about|keep in mind|note that|observe that|notice that|realize that|understand that|recognize that|acknowledge that|accept that|assume that|suppose that|imagine that|pretend that|act as if|behave as if|respond as if|treat as|regard as|view as|see as|consider as|think of as|look at as|approach as|handle as|deal with as|work with as|use as|employ as|utilize as|leverage as)\s+\w+', prompt.lower())
        
        patterns.extend(imperatives)
        
        # Question patterns
        questions = re.findall(r'\b(?:what|how|when|where|why|which|who|whose|whom|can you|could you|would you|will you|do you|did you|have you|has it|is it|are they|was it|were they)\b', prompt.lower())
        
        patterns.extend(questions)
        
        # Conditional patterns
        conditionals = re.findall(r'\b(?:if|when|unless|provided that|given that|assuming that|in case|should|were|had)\b', prompt.lower())
        
        patterns.extend(conditionals)
        
        return patterns
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\b\w+\b', text)


class PromptStyleAnalyzer:
    """Analyze prompt style consistency across multiple dimensions."""
    
    def analyze_consistency(self, prompt1: str, prompt2: str) -> float:
        """Analyze style consistency across multiple dimensions."""
        
        try:
            # Style dimension analyses
            formality_consistency = self._analyze_formality_consistency(prompt1, prompt2)
            tone_consistency = self._analyze_tone_consistency(prompt1, prompt2)
            structure_consistency = self._analyze_structure_consistency(prompt1, prompt2)
            vocabulary_consistency = self._analyze_vocabulary_consistency(prompt1, prompt2)
            length_consistency = self._analyze_length_consistency(prompt1, prompt2)
            
            # Weighted combination
            consistency = (
                0.25 * formality_consistency +
                0.25 * tone_consistency +
                0.2 * structure_consistency +
                0.2 * vocabulary_consistency +
                0.1 * length_consistency
            )
            
            return max(0.0, min(1.0, consistency))
            
        except Exception as e:
            logger.debug(f"Style analysis failed: {e}")
            return 0.5
    
    def _analyze_formality_consistency(self, prompt1: str, prompt2: str) -> float:
        """Analyze consistency in formality level."""
        
        formality1 = self._calculate_formality_score(prompt1)
        formality2 = self._calculate_formality_score(prompt2)
        
        # Consistency based on similarity of formality scores
        consistency = 1.0 - abs(formality1 - formality2)
        
        return max(0.0, consistency)
    
    def _analyze_tone_consistency(self, prompt1: str, prompt2: str) -> float:
        """Analyze consistency in tone."""
        
        tone1 = self._identify_tone(prompt1)
        tone2 = self._identify_tone(prompt2)
        
        # Simple tone matching
        if tone1 == tone2:
            return 1.0
        
        # Related tones have partial consistency
        related_tones = {
            'instructional': ['educational', 'explanatory'],
            'conversational': ['friendly', 'casual'],
            'professional': ['formal', 'business'],
            'analytical': ['technical', 'scientific']
        }
        
        for base_tone, related in related_tones.items():
            if (tone1 == base_tone and tone2 in related) or (tone2 == base_tone and tone1 in related):
                return 0.7
            if tone1 in related and tone2 in related:
                return 0.8
        
        return 0.3  # Different tones
    
    def _analyze_structure_consistency(self, prompt1: str, prompt2: str) -> float:
        """Analyze consistency in structural patterns."""
        
        structure1 = self._analyze_structure(prompt1)
        structure2 = self._analyze_structure(prompt2)
        
        # Compare structural elements
        consistency_factors = []
        
        # Sentence count similarity
        sent_count_diff = abs(structure1['sentence_count'] - structure2['sentence_count'])
        sent_consistency = max(0.0, 1.0 - sent_count_diff / max(structure1['sentence_count'], structure2['sentence_count'], 1))
        consistency_factors.append(sent_consistency)
        
        # Paragraph structure similarity
        para_consistency = 1.0 if structure1['has_paragraphs'] == structure2['has_paragraphs'] else 0.5
        consistency_factors.append(para_consistency)
        
        # List/enumeration similarity
        list_consistency = 1.0 if structure1['has_lists'] == structure2['has_lists'] else 0.7
        consistency_factors.append(list_consistency)
        
        # Example patterns similarity
        example_consistency = 1.0 if structure1['has_examples'] == structure2['has_examples'] else 0.8
        consistency_factors.append(example_consistency)
        
        return np.mean(consistency_factors)
    
    def _analyze_vocabulary_consistency(self, prompt1: str, prompt2: str) -> float:
        """Analyze consistency in vocabulary sophistication and domain terms."""
        
        vocab1 = self._analyze_vocabulary(prompt1)
        vocab2 = self._analyze_vocabulary(prompt2)
        
        # Compare vocabulary sophistication
        sophistication_diff = abs(vocab1['sophistication'] - vocab2['sophistication'])
        sophistication_consistency = max(0.0, 1.0 - sophistication_diff)
        
        # Compare domain vocabulary overlap
        domain_overlap = len(set(vocab1['domain_terms']) & set(vocab2['domain_terms']))
        total_domain_terms = len(set(vocab1['domain_terms']) | set(vocab2['domain_terms']))
        domain_consistency = domain_overlap / max(total_domain_terms, 1)
        
        # Technical terminology consistency
        tech_consistency = 1.0 if vocab1['has_technical_terms'] == vocab2['has_technical_terms'] else 0.6
        
        return (0.4 * sophistication_consistency + 0.4 * domain_consistency + 0.2 * tech_consistency)
    
    def _analyze_length_consistency(self, prompt1: str, prompt2: str) -> float:
        """Analyze consistency in prompt length."""
        
        len1 = len(prompt1)
        len2 = len(prompt2)
        
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Length similarity
        length_ratio = min(len1, len2) / max(len1, len2)
        
        return length_ratio
    
    def _calculate_formality_score(self, prompt: str) -> float:
        """Calculate formality score for prompt."""
        
        formality_indicators = {
            'formal': ['please', 'kindly', 'ensure', 'provide', 'utilize', 'implement', 'demonstrate', 'facilitate', 'consequently', 'furthermore', 'therefore', 'however', 'moreover', 'nevertheless'],
            'informal': ['hey', 'ok', 'yeah', 'nope', 'gonna', 'wanna', 'kinda', 'sorta', 'really', 'pretty', 'quite', 'super', 'awesome', 'cool']
        }
        
        text_lower = prompt.lower()
        
        formal_count = sum(1 for word in formality_indicators['formal'] if word in text_lower)
        informal_count = sum(1 for word in formality_indicators['informal'] if word in text_lower)
        
        total_words = len(prompt.split())
        
        if total_words == 0:
            return 0.5
        
        formal_ratio = formal_count / total_words
        informal_ratio = informal_count / total_words
        
        # Formality score: higher = more formal
        formality_score = 0.5 + (formal_ratio - informal_ratio) * 5
        
        return max(0.0, min(1.0, formality_score))
    
    def _identify_tone(self, prompt: str) -> str:
        """Identify the dominant tone of the prompt."""
        
        text_lower = prompt.lower()
        
        tone_indicators = {
            'instructional': ['step', 'first', 'then', 'next', 'finally', 'instruction', 'guide', 'how to', 'follow', 'complete'],
            'conversational': ['you', 'your', 'let\'s', 'we', 'our', 'together', 'chat', 'talk', 'discuss'],
            'professional': ['business', 'company', 'organization', 'professional', 'corporate', 'industry', 'market'],
            'analytical': ['analyze', 'examine', 'evaluate', 'assess', 'investigate', 'research', 'study', 'data', 'evidence'],
            'creative': ['create', 'generate', 'imagine', 'design', 'craft', 'innovative', 'original', 'unique', 'artistic'],
            'technical': ['implement', 'configure', 'execute', 'deploy', 'algorithm', 'function', 'parameter', 'variable']
        }
        
        tone_scores = {}
        for tone, indicators in tone_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            tone_scores[tone] = score
        
        if not tone_scores or max(tone_scores.values()) == 0:
            return 'neutral'
        
        return max(tone_scores.keys(), key=lambda x: tone_scores[x])
    
    def _analyze_structure(self, prompt: str) -> Dict[str, Any]:
        """Analyze structural elements of prompt."""
        
        sentences = prompt.split('.')
        paragraphs = prompt.split('\n\n')
        
        return {
            'sentence_count': len([s for s in sentences if s.strip()]),
            'paragraph_count': len([p for p in paragraphs if p.strip()]),
            'has_paragraphs': len(paragraphs) > 1,
            'has_lists': bool(re.search(r'[•\-\*]\s|^\d+\.\s|\n\s*[-*]\s', prompt, re.MULTILINE)),
            'has_examples': bool(re.search(r'example|for instance|such as|like|e\.g\.|including', prompt.lower())),
            'has_questions': '?' in prompt,
            'has_imperatives': bool(re.search(r'\b(?:please|ensure|make sure|do not|don\'t)\b', prompt.lower()))
        }
    
    def _analyze_vocabulary(self, prompt: str) -> Dict[str, Any]:
        """Analyze vocabulary characteristics."""
        
        words = re.findall(r'\b\w+\b', prompt.lower())
        
        # Sophistication based on word length and complexity
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        sophistication = min(1.0, (avg_word_length - 3) / 5)  # Normalize to [0, 1]
        
        # Domain-specific terms (simplified)
        domain_terms = [word for word in words if len(word) > 6 and word not in ['through', 'between', 'within', 'without', 'during', 'before', 'after']]
        
        # Technical terms
        technical_indicators = ['algorithm', 'function', 'parameter', 'variable', 'implementation', 'configuration', 'optimization', 'analysis', 'processing', 'computation']
        has_technical_terms = any(term in prompt.lower() for term in technical_indicators)
        
        return {
            'sophistication': sophistication,
            'domain_terms': domain_terms,
            'has_technical_terms': has_technical_terms,
            'word_count': len(words),
            'unique_word_ratio': len(set(words)) / max(len(words), 1)
        }


class InputOutputAnalyzer:  
    """Analyze input/output format compatibility between modules."""
    
    def check_format_compatibility(
        self,
        module1: Any,  # LanguageModule type
        module2: Any,  # LanguageModule type
        context: InteractionContext
    ) -> float:
        """Check input/output format compatibility."""
        
        try:
            # Extract expected input/output formats from prompts
            format1 = self._extract_io_format(module1)
            format2 = self._extract_io_format(module2)
            
            # Analyze format compatibility
            input_compatibility = self._analyze_input_compatibility(format1, format2, context)
            output_compatibility = self._analyze_output_compatibility(format1, format2, context)
            data_type_compatibility = self._analyze_data_type_compatibility(format1, format2)
            
            # Weighted combination
            compatibility = (
                0.4 * input_compatibility +
                0.4 * output_compatibility +
                0.2 * data_type_compatibility
            )
            
            return max(0.0, min(1.0, compatibility))
            
        except Exception as e:
            logger.debug(f"I/O compatibility analysis failed: {e}")
            return 0.7  # Default compatible
    
    def _extract_io_format(self, module: Any) -> Dict[str, Any]:
        """Extract input/output format expectations from module."""
        
        prompt = getattr(module, 'prompt', '')
        
        # Analyze prompt for format expectations
        format_info = {
            'input_type': 'text',  # Default
            'output_type': 'text',  # Default
            'structured_input': False,
            'structured_output': False,
            'format_requirements': []
        }
        
        # Look for structured input/output indicators
        if re.search(r'json|yaml|xml|csv|table|list|array', prompt.lower()):
            format_info['structured_output'] = True
        
        if re.search(r'format.*as|output.*format|return.*as|provide.*in.*format', prompt.lower()):
            format_info['structured_output'] = True
        
        # Extract specific format requirements
        format_matches = re.findall(r'format.*?(?:json|yaml|xml|csv|table|list|array|markdown|html)', prompt.lower())
        format_info['format_requirements'] = format_matches
        
        return format_info
    
    def _analyze_input_compatibility(
        self,
        format1: Dict[str, Any],
        format2: Dict[str, Any],
        context: InteractionContext
    ) -> float:
        """Analyze input format compatibility."""
        
        # If modules are not directly connected, input compatibility is less critical
        if not self._modules_connected(format1, format2, context):
            return 0.8
        
        # Check input type compatibility
        if format1['input_type'] == format2['input_type']:
            return 1.0
        
        # Compatible input types
        compatible_types = {
            ('text', 'string'): 0.9,
            ('json', 'structured'): 0.9,
            ('list', 'array'): 1.0
        }
        
        type_pair = (format1['input_type'], format2['input_type'])
        return compatible_types.get(type_pair, compatible_types.get((type_pair[1], type_pair[0]), 0.5))
    
    def _analyze_output_compatibility(
        self,
        format1: Dict[str, Any],
        format2: Dict[str, Any],
        context: InteractionContext
    ) -> float:
        """Analyze output format compatibility."""
        
        # Similar logic to input compatibility
        if format1['output_type'] == format2['output_type']:
            return 1.0
        
        # Check if structured outputs are compatible
        if format1['structured_output'] and format2['structured_output']:
            return 0.8
        elif format1['structured_output'] != format2['structured_output']:
            return 0.6  # One structured, one unstructured
        
        return 0.7  # Default compatibility
    
    def _analyze_data_type_compatibility(
        self,
        format1: Dict[str, Any],
        format2: Dict[str, Any]
    ) -> float:
        """Analyze data type compatibility."""
        
        # Compare format requirements
        req1 = set(format1['format_requirements'])
        req2 = set(format2['format_requirements'])
        
        if not req1 or not req2:
            return 0.8  # No specific requirements
        
        # Calculate overlap
        overlap = len(req1 & req2)
        total = len(req1 | req2)
        
        return overlap / max(total, 1)
    
    def _modules_connected(
        self,
        format1: Dict[str, Any],
        format2: Dict[str, Any],
        context: InteractionContext
    ) -> bool:
        """Check if modules are directly connected in workflow."""
        
        # Simplified - would analyze actual data flow
        return True  # Assume connected for now


class PerformanceCorrelationAnalyzer:
    """Analyze performance correlation between modules."""
    
    def analyze_correlation(
        self,
        module1: Any,
        module2: Any,
        context: InteractionContext
    ) -> float:
        """Analyze performance correlation between modules."""
        
        try:
            module1_id = getattr(module1, 'id', 'module1')
            module2_id = getattr(module2, 'id', 'module2')
            
            # Get performance metrics for both modules
            perf1 = context.performance_metrics.get(module1_id, {})
            perf2 = context.performance_metrics.get(module2_id, {})
            
            if not perf1 or not perf2:
                return 0.5  # Default neutral correlation
            
            # Calculate correlation across different metrics
            correlations = []
            
            common_metrics = set(perf1.keys()) & set(perf2.keys())
            
            for metric in common_metrics:
                if isinstance(perf1[metric], (list, tuple)) and isinstance(perf2[metric], (list, tuple)):
                    # Time series correlation
                    correlation = self._calculate_time_series_correlation(perf1[metric], perf2[metric])
                    correlations.append(correlation)
                elif isinstance(perf1[metric], (int, float)) and isinstance(perf2[metric], (int, float)):
                    # Single value - use compatibility as proxy
                    compatibility = 1.0 - abs(perf1[metric] - perf2[metric]) / max(abs(perf1[metric]), abs(perf2[metric]), 1)
                    correlations.append(compatibility)
            
            if not correlations:
                return 0.5
            
            return np.mean(correlations)
            
        except Exception as e:
            logger.debug(f"Performance correlation analysis failed: {e}")
            return 0.5
    
    def _calculate_time_series_correlation(self, series1: List[float], series2: List[float]) -> float:
        """Calculate correlation between two time series."""
        
        if len(series1) != len(series2) or len(series1) < 2:
            return 0.5
        
        try:
            correlation = np.corrcoef(series1, series2)[0, 1]
            if np.isnan(correlation):
                return 0.5
            
            # Convert correlation to compatibility score (abs value, higher is better)
            return abs(correlation)
            
        except Exception:
            return 0.5


class SophisticatedCompatibilityAnalyzer:
    """Deep analysis of module compatibility using multiple dimensions."""
    
    def __init__(self):
        self.semantic_analyzer = SemanticSimilarityAnalyzer()
        self.style_analyzer = PromptStyleAnalyzer()
        self.io_analyzer = InputOutputAnalyzer()
        self.performance_analyzer = PerformanceCorrelationAnalyzer()
        
        # Dimension weights (can be adaptive)
        self.dimension_weights = {
            CompatibilityDimension.SEMANTIC: 0.25,
            CompatibilityDimension.STYLE: 0.25,
            CompatibilityDimension.INPUT_OUTPUT: 0.25,
            CompatibilityDimension.PERFORMANCE: 0.15,
            CompatibilityDimension.LOGICAL_FLOW: 0.1
        }
    
    def analyze_module_compatibility(
        self,
        module1: Any,  # LanguageModule type
        module2: Any,  # LanguageModule type
        interaction_context: InteractionContext
    ) -> CompatibilityAnalysis:
        """Comprehensive compatibility analysis."""
        
        try:
            # Dimension 1: Semantic Compatibility
            semantic_score = self.semantic_analyzer.calculate_similarity(
                getattr(module1, 'prompt', ''),
                getattr(module2, 'prompt', '')
            )
            
            # Dimension 2: Style Consistency
            style_score = self.style_analyzer.analyze_consistency(
                getattr(module1, 'prompt', ''),
                getattr(module2, 'prompt', '')
            )
            
            # Dimension 3: Input/Output Format Compatibility
            io_score = self.io_analyzer.check_format_compatibility(
                module1, module2, interaction_context
            )
            
            # Dimension 4: Performance Correlation
            perf_score = self.performance_analyzer.analyze_correlation(
                module1, module2, interaction_context
            )
            
            # Dimension 5: Logical Flow Compatibility
            flow_score = self._analyze_logical_flow_compatibility(
                module1, module2, interaction_context
            )
            
            # Calculate weighted overall score
            overall_score = (
                self.dimension_weights[CompatibilityDimension.SEMANTIC] * semantic_score +
                self.dimension_weights[CompatibilityDimension.STYLE] * style_score +
                self.dimension_weights[CompatibilityDimension.INPUT_OUTPUT] * io_score +
                self.dimension_weights[CompatibilityDimension.PERFORMANCE] * perf_score +
                self.dimension_weights[CompatibilityDimension.LOGICAL_FLOW] * flow_score
            )
            
            # Generate explanation
            explanation = self._generate_compatibility_explanation(
                semantic_score, style_score, io_score, perf_score, flow_score
            )
            
            # Calculate confidence based on data quality
            confidence = self._calculate_analysis_confidence(
                module1, module2, interaction_context
            )
            
            return CompatibilityAnalysis(
                semantic_compatibility=semantic_score,
                style_consistency=style_score,
                io_compatibility=io_score,
                performance_correlation=perf_score,
                logical_flow=flow_score,
                overall_score=overall_score,
                explanation=explanation,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Compatibility analysis failed: {e}")
            # Return default compatibility analysis
            return CompatibilityAnalysis(
                semantic_compatibility=0.5,
                style_consistency=0.5,
                io_compatibility=0.7,
                performance_correlation=0.5,
                logical_flow=0.6,
                overall_score=0.56,
                explanation="Analysis failed - using default compatibility scores",
                confidence=0.3
            )
    
    def _analyze_logical_flow_compatibility(
        self,
        module1: Any,
        module2: Any,
        interaction_context: InteractionContext
    ) -> float:
        """Analyze if modules work well together in logical flow."""
        
        try:
            # Get module positions in workflow
            module1_id = getattr(module1, 'id', 'module1')
            module2_id = getattr(module2, 'id', 'module2')
            
            position1 = interaction_context.workflow_position.get(module1_id, 0)
            position2 = interaction_context.workflow_position.get(module2_id, 1)
            
            # Analyze logical relationship based on positions
            if abs(position1 - position2) == 1:
                # Adjacent modules - check direct compatibility
                return self._analyze_adjacent_compatibility(module1, module2, interaction_context)
            else:
                # Non-adjacent - check overall workflow coherence
                return self._analyze_workflow_coherence(module1, module2, interaction_context)
                
        except Exception as e:
            logger.debug(f"Logical flow analysis failed: {e}")
            return 0.6
    
    def _analyze_adjacent_compatibility(
        self,
        module1: Any,
        module2: Any,
        interaction_context: InteractionContext
    ) -> float:
        """Analyze compatibility between adjacent modules."""
        
        # Check if output of module1 is suitable input for module2
        prompt1 = getattr(module1, 'prompt', '')
        prompt2 = getattr(module2, 'prompt', '')
        
        # Look for handoff patterns
        handoff_indicators = [
            ('generate', 'analyze'),
            ('extract', 'summarize'),
            ('classify', 'rank'),
            ('process', 'format'),
            ('create', 'review')
        ]
        
        for pattern1, pattern2 in handoff_indicators:
            if pattern1 in prompt1.lower() and pattern2 in prompt2.lower():
                return 0.9  # Good handoff pattern
        
        return 0.6  # Default adjacent compatibility
    
    def _analyze_workflow_coherence(
        self,
        module1: Any,
        module2: Any,
        interaction_context: InteractionContext
    ) -> float:
        """Analyze overall workflow coherence."""
        
        # Simplified - would analyze end-to-end workflow compatibility
        return 0.7
    
    def _generate_compatibility_explanation(
        self,
        semantic: float,
        style: float,
        io: float,
        performance: float,
        flow: float
    ) -> str:
        """Generate human-readable explanation of compatibility analysis."""
        
        explanations = []
        
        if semantic > 0.8:
            explanations.append("Strong semantic similarity")
        elif semantic < 0.3:
            explanations.append("Low semantic similarity")
        
        if style > 0.8:
            explanations.append("Consistent style")
        elif style < 0.3:
            explanations.append("Style inconsistency")
        
        if io > 0.8:
            explanations.append("Compatible I/O formats")
        elif io < 0.4:
            explanations.append("I/O format mismatch")
        
        if performance > 0.7:
            explanations.append("Correlated performance")
        elif performance < 0.3:
            explanations.append("Uncorrelated performance")
        
        if flow > 0.8:
            explanations.append("Good logical flow")
        elif flow < 0.4:
            explanations.append("Poor logical flow")
        
        if not explanations:
            explanations.append("Moderate compatibility across dimensions")
        
        return "; ".join(explanations)
    
    def _calculate_analysis_confidence(
        self,
        module1: Any,
        module2: Any,
        interaction_context: InteractionContext
    ) -> float:
        """Calculate confidence in compatibility analysis."""
        
        confidence_factors = []
        
        # Factor 1: Availability of prompt text
        prompt1 = getattr(module1, 'prompt', '')
        prompt2 = getattr(module2, 'prompt', '')
        
        if prompt1 and prompt2:
            confidence_factors.append(1.0)
        elif prompt1 or prompt2:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.1)
        
        # Factor 2: Availability of performance data
        module1_id = getattr(module1, 'id', 'module1')
        module2_id = getattr(module2, 'id', 'module2')
        
        if (module1_id in interaction_context.performance_metrics and 
            module2_id in interaction_context.performance_metrics):
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.4)
        
        # Factor 3: Execution history availability
        if (module1_id in interaction_context.execution_history and
            module2_id in interaction_context.execution_history):
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
        
        return np.mean(confidence_factors) 