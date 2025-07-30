"""Strategic minibatch sampling replacing random sampling heuristics."""

from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    """Different sampling strategies available."""
    DIFFICULTY_AWARE = "difficulty_aware"
    DIVERSITY_BASED = "diversity_based"
    ERROR_FOCUSED = "error_focused"
    UNCERTAINTY_BASED = "uncertainty_based"


@dataclass
class SamplingContext:
    """Context for strategic minibatch sampling."""
    current_candidate: Any  # Candidate type
    sampling_strategy: str = "balanced"  # balanced, aggressive, conservative
    exploration_factor: float = 0.3
    difficulty_target: str = "medium"  # easy, medium, hard, mixed
    diversity_threshold: float = 0.7


@dataclass
class InstanceAnalysis:
    """Analysis of a training instance."""
    instance_id: int
    difficulty_score: float
    diversity_score: float
    error_likelihood: float
    uncertainty_score: float
    feature_vector: Optional[np.ndarray] = None


class DifficultyAnalyzer:
    """Analyze difficulty of instances for current candidate."""
    
    def __init__(self):
        self.difficulty_cache: Dict[Tuple[str, int], float] = {}
        self.feature_extractors = []
        
    def predict_difficulty(
        self,
        instance: Dict[str, Any],
        instance_id: int,
        candidate: Any
    ) -> float:
        """Predict how difficult this instance is for the current candidate."""
        
        cache_key = (candidate.id, instance_id)
        if cache_key in self.difficulty_cache:
            return self.difficulty_cache[cache_key]
        
        try:
            # Extract features for difficulty prediction
            features = self._extract_difficulty_features(instance)
            
            # Method 1: Length-based difficulty heuristic
            length_difficulty = self._calculate_length_difficulty(instance)
            
            # Method 2: Complexity-based difficulty
            complexity_difficulty = self._calculate_complexity_difficulty(instance)
            
            # Method 3: Historical performance-based difficulty
            historical_difficulty = self._calculate_historical_difficulty(
                instance_id, candidate
            )
            
            # Method 4: Content-based difficulty
            content_difficulty = self._calculate_content_difficulty(instance)
            
            # Combine difficulty measures
            combined_difficulty = (
                0.25 * length_difficulty +
                0.25 * complexity_difficulty +
                0.3 * historical_difficulty +
                0.2 * content_difficulty
            )
            
            # Cache result
            self.difficulty_cache[cache_key] = combined_difficulty
            
            return combined_difficulty
            
        except Exception as e:
            logger.debug(f"Difficulty prediction failed for instance {instance_id}: {e}")
            return 0.5  # Default medium difficulty
    
    def _extract_difficulty_features(self, instance: Dict[str, Any]) -> np.ndarray:
        """Extract features relevant to difficulty assessment."""
        
        features = []
        
        # Text-based features
        if 'text' in instance:
            text = str(instance['text'])
            features.extend([
                len(text),
                len(text.split()),
                len(text.split('\n')),
                text.count('?'),
                text.count('!'),
                text.count('.'),
                len(set(text.lower().split())),  # Unique words
            ])
        
        # Task-specific features would go here based on the domain
        
        return np.array(features, dtype=float) if features else np.array([0.0])
    
    def _calculate_length_difficulty(self, instance: Dict[str, Any]) -> float:
        """Calculate difficulty based on input length."""
        
        if 'text' in instance:
            text = str(instance['text'])
            length = len(text)
            
            # Normalize length to difficulty score
            # Assume longer texts are generally more difficult
            if length < 100:
                return 0.2  # Easy
            elif length < 500:
                return 0.5  # Medium
            elif length < 1000:
                return 0.7  # Hard
            else:
                return 0.9  # Very hard
        
        return 0.5  # Default
    
    def _calculate_complexity_difficulty(self, instance: Dict[str, Any]) -> float:
        """Calculate difficulty based on content complexity."""
        
        if 'text' in instance:
            text = str(instance['text'])
            
            # Complexity indicators
            complexity_indicators = 0
            
            # Sentence complexity
            sentences = text.split('.')
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
            if avg_sentence_length > 20:
                complexity_indicators += 1
            
            # Vocabulary complexity
            words = text.lower().split()
            unique_word_ratio = len(set(words)) / max(len(words), 1)
            if unique_word_ratio > 0.7:
                complexity_indicators += 1
            
            # Special characters/formatting
            if any(char in text for char in ['(', ')', '[', ']', '{', '}', '|']):
                complexity_indicators += 1
            
            # Numbers and formulas
            if any(char.isdigit() for char in text):
                complexity_indicators += 1
            
            # Normalize to [0, 1]
            return min(1.0, complexity_indicators / 4.0)
        
        return 0.5
    
    def _calculate_historical_difficulty(self, instance_id: int, candidate: Any) -> float:
        """Calculate difficulty based on historical performance."""
        
        # Check if candidate has performed on this instance before
        if hasattr(candidate, 'scores') and isinstance(candidate.scores, dict):
            if instance_id in candidate.scores:
                score = candidate.scores[instance_id]
                # Low score = high difficulty
                return max(0.0, 1.0 - score)
        
        # Check historical performance on similar instances (simplified)
        # In practice, would use similarity-based lookup
        if hasattr(candidate, 'scores') and candidate.scores:
            avg_score = np.mean(list(candidate.scores.values()))
            return max(0.0, 1.0 - avg_score)
        
        return 0.5  # Default medium difficulty
    
    def _calculate_content_difficulty(self, instance: Dict[str, Any]) -> float:
        """Calculate difficulty based on content analysis."""
        
        if 'text' in instance:
            text = str(instance['text']).lower()
            
            # Domain-specific difficulty keywords
            difficult_keywords = [
                'complex', 'complicated', 'intricate', 'sophisticated', 
                'advanced', 'technical', 'specialized', 'detailed'
            ]
            
            easy_keywords = [
                'simple', 'basic', 'easy', 'straightforward', 'clear', 'obvious'
            ]
            
            difficult_count = sum(1 for word in difficult_keywords if word in text)
            easy_count = sum(1 for word in easy_keywords if word in text)
            
            # Calculate relative difficulty
            if difficult_count + easy_count > 0:
                return difficult_count / (difficult_count + easy_count)
            
        return 0.5


class DiversityAnalyzer:
    """Analyze diversity to ensure representative minibatch selection."""
    
    def __init__(self):
        self.feature_cache: Dict[int, np.ndarray] = {}
        self.cluster_cache: Dict[str, Any] = {}
        
    def calculate_diversity_scores(
        self,
        instances: List[Dict[str, Any]],
        already_selected: List[int] = None
    ) -> Dict[int, float]:
        """Calculate diversity scores for all instances."""
        
        already_selected = already_selected or []
        
        # Extract features for all instances
        features_list = []
        instance_ids = []
        
        for i, instance in enumerate(instances):
            features = self._extract_features(instance, i)
            if features is not None:
                features_list.append(features)
                instance_ids.append(i)
        
        if len(features_list) < 2:
            return {i: 1.0 for i in range(len(instances))}
        
        # Convert to numpy array
        features_matrix = np.array(features_list)
        
        # Calculate pairwise distances
        distances = pairwise_distances(features_matrix, metric='cosine')
        
        diversity_scores = {}
        for i, instance_id in enumerate(instance_ids):
            if instance_id in already_selected:
                # Already selected instances have zero diversity benefit
                diversity_scores[instance_id] = 0.0
            else:
                # Calculate diversity relative to already selected instances
                if already_selected:
                    selected_indices = [instance_ids.index(sid) for sid in already_selected 
                                     if sid in instance_ids]
                    if selected_indices:
                        min_distance = np.min(distances[i, selected_indices])
                        diversity_scores[instance_id] = min_distance
                    else:
                        diversity_scores[instance_id] = 1.0
                else:
                    # No instances selected yet - use average distance to all others
                    avg_distance = np.mean(distances[i, :])
                    diversity_scores[instance_id] = avg_distance
        
        # Normalize scores
        if diversity_scores:
            max_score = max(diversity_scores.values())
            if max_score > 0:
                diversity_scores = {k: v/max_score for k, v in diversity_scores.items()}
        
        return diversity_scores
    
    def _extract_features(self, instance: Dict[str, Any], instance_id: int) -> Optional[np.ndarray]:
        """Extract features for diversity calculation."""
        
        if instance_id in self.feature_cache:
            return self.feature_cache[instance_id]
        
        try:
            features = []
            
            # Text-based features
            if 'text' in instance:
                text = str(instance['text'])
                
                # Simple bag-of-words features (in practice, would use embeddings)
                words = text.lower().split()
                
                # Feature categories
                features.extend([
                    len(text),
                    len(words),
                    len(set(words)),
                    np.mean([len(word) for word in words]) if words else 0,
                    text.count('?'),
                    text.count('!'),
                    text.count('.'),
                ])
                
                # Simple n-gram features (first 10 most common words)  
                word_counts = defaultdict(int)
                for word in words[:100]:  # Limit for performance
                    word_counts[word] += 1
                
                common_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to']
                for word in common_words:
                    features.append(word_counts.get(word, 0))
            
            if features:
                feature_array = np.array(features, dtype=float)
                self.feature_cache[instance_id] = feature_array
                return feature_array
                
        except Exception as e:
            logger.debug(f"Feature extraction failed for instance {instance_id}: {e}")
        
        return None
    
    def cluster_instances(
        self,
        instances: List[Dict[str, Any]],
        n_clusters: int = 5
    ) -> Dict[int, int]:
        """Cluster instances for diversity sampling."""
        
        # Extract features
        features_list = []
        instance_ids = []
        
        for i, instance in enumerate(instances):
            features = self._extract_features(instance, i)
            if features is not None:
                features_list.append(features)
                instance_ids.append(i)
        
        if len(features_list) < n_clusters:
            # Not enough instances for clustering
            return {i: i % n_clusters for i in range(len(instances))}
        
        try:
            # Perform K-means clustering
            features_matrix = np.array(features_list)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_matrix)
            
            # Map back to instance IDs
            cluster_mapping = {}
            for i, instance_id in enumerate(instance_ids):
                cluster_mapping[instance_id] = cluster_labels[i]
            
            # Assign remaining instances to random clusters
            for i in range(len(instances)):
                if i not in cluster_mapping:
                    cluster_mapping[i] = i % n_clusters
            
            return cluster_mapping
            
        except Exception as e:
            logger.debug(f"Clustering failed: {e}")
            return {i: i % n_clusters for i in range(len(instances))}


class ErrorFocusedSampler:
    """Sample instances that are likely to reveal errors or weaknesses."""
    
    def __init__(self):
        self.error_patterns: Dict[str, List[int]] = defaultdict(list)
        
    def calculate_error_likelihood(
        self,
        instances: List[Dict[str, Any]],
        candidate: Any
    ) -> Dict[int, float]:
        """Calculate likelihood of error for each instance."""
        
        error_scores = {}
        
        for i, instance in enumerate(instances):
            # Method 1: Historical error patterns
            historical_score = self._get_historical_error_score(i, candidate)
            
            # Method 2: Content-based error indicators
            content_score = self._get_content_error_score(instance)
            
            # Method 3: Similarity to previous error cases
            similarity_score = self._get_similarity_error_score(i, instance, candidate)
            
            # Combine scores
            error_scores[i] = (
                0.4 * historical_score +
                0.3 * content_score +
                0.3 * similarity_score
            )
        
        return error_scores
    
    def _get_historical_error_score(self, instance_id: int, candidate: Any) -> float:
        """Get error score based on historical performance."""
        
        candidate_id = getattr(candidate, 'id', 'unknown')
        
        if candidate_id in self.error_patterns:
            if instance_id in self.error_patterns[candidate_id]:
                return 1.0  # High likelihood - previously failed
            
        # Check performance if available
        if hasattr(candidate, 'scores') and isinstance(candidate.scores, dict):
            if instance_id in candidate.scores:
                score = candidate.scores[instance_id]
                # Low score suggests high error likelihood
                return max(0.0, 1.0 - score)
        
        return 0.5  # Default medium likelihood
    
    def _get_content_error_score(self, instance: Dict[str, Any]) -> float:
        """Calculate error likelihood based on content analysis."""
        
        if 'text' in instance:
            text = str(instance['text']).lower()
            
            # Error-prone content indicators
            error_indicators = [
                'ambiguous', 'unclear', 'complex', 'multiple', 'various',
                'exception', 'special case', 'edge case', 'unusual'
            ]
            
            indicator_count = sum(1 for indicator in error_indicators if indicator in text)
            
            # Normalize
            return min(1.0, indicator_count / 3.0)
        
        return 0.3
    
    def _get_similarity_error_score(self, instance_id: int, instance: Dict[str, Any], candidate: Any) -> float:
        """Calculate error likelihood based on similarity to previous errors."""
        
        # Simplified implementation - would use more sophisticated similarity
        candidate_id = getattr(candidate, 'id', 'unknown')
        
        if candidate_id not in self.error_patterns or not self.error_patterns[candidate_id]:
            return 0.3
        
        # Check if this instance is similar to previous error cases
        # This is a placeholder - real implementation would use semantic similarity
        
        return 0.5  # Default
    
    def record_error(self, instance_id: int, candidate_id: str) -> None:
        """Record an error for future sampling."""
        self.error_patterns[candidate_id].append(instance_id)
        
        # Keep error history manageable
        if len(self.error_patterns[candidate_id]) > 50:
            self.error_patterns[candidate_id] = self.error_patterns[candidate_id][-50:]


class StrategicMinibatchSampler:
    """Intelligent minibatch sampling for optimal learning.
    
    Replaces random sampling with strategic selection using:
    1. Difficulty-aware sampling - target optimal challenge level
    2. Diversity-based sampling - ensure representative coverage
    3. Error-focused sampling - emphasize likely failure cases
    4. Uncertainty-based sampling - target high-uncertainty regions
    """
    
    def __init__(self):
        self.difficulty_analyzer = DifficultyAnalyzer()
        self.diversity_analyzer = DiversityAnalyzer()
        self.error_sampler = ErrorFocusedSampler()
        self.sampling_history: List[List[int]] = []
        
    def sample_minibatch(
        self,
        training_dataset: List[Dict[str, Any]],
        minibatch_size: int,
        context: SamplingContext
    ) -> List[Dict[str, Any]]:
        """Sample minibatch using strategic selection."""
        
        if minibatch_size >= len(training_dataset):
            return training_dataset
        
        if minibatch_size <= 0:
            return []
        
        # Analyze all instances
        instance_analyses = self._analyze_all_instances(training_dataset, context)
        
        # Apply strategic sampling based on context
        if context.sampling_strategy == "aggressive":
            selected_indices = self._aggressive_sampling(instance_analyses, minibatch_size)
        elif context.sampling_strategy == "conservative":
            selected_indices = self._conservative_sampling(instance_analyses, minibatch_size)
        else:  # balanced
            selected_indices = self._balanced_sampling(instance_analyses, minibatch_size)
        
        # Record sampling for learning
        self.sampling_history.append(selected_indices)
        
        # Return selected instances
        return [training_dataset[i] for i in selected_indices]
    
    def _analyze_all_instances(
        self,
        training_dataset: List[Dict[str, Any]],
        context: SamplingContext
    ) -> List[InstanceAnalysis]:
        """Analyze all instances for strategic sampling."""
        
        analyses = []
        
        # Calculate difficulty scores
        difficulty_scores = {}
        for i, instance in enumerate(training_dataset):
            difficulty = self.difficulty_analyzer.predict_difficulty(
                instance, i, context.current_candidate
            )
            difficulty_scores[i] = difficulty
        
        # Calculate diversity scores
        diversity_scores = self.diversity_analyzer.calculate_diversity_scores(training_dataset)
        
        # Calculate error likelihood scores
        error_scores = self.error_sampler.calculate_error_likelihood(
            training_dataset, context.current_candidate
        )
        
        # Calculate uncertainty scores (simplified)
        uncertainty_scores = self._calculate_uncertainty_scores(
            training_dataset, context.current_candidate
        )
        
        # Combine into analyses
        for i in range(len(training_dataset)):
            analysis = InstanceAnalysis(
                instance_id=i,
                difficulty_score=difficulty_scores.get(i, 0.5),
                diversity_score=diversity_scores.get(i, 0.5),
                error_likelihood=error_scores.get(i, 0.5),
                uncertainty_score=uncertainty_scores.get(i, 0.5)
            )
            analyses.append(analysis)
        
        return analyses
    
    def _balanced_sampling(
        self,
        analyses: List[InstanceAnalysis],
        minibatch_size: int
    ) -> List[int]:
        """Balanced sampling across all strategies."""
        
        # Allocate minibatch slots to different strategies
        difficulty_slots = max(1, minibatch_size // 3)
        diversity_slots = max(1, minibatch_size // 3)
        error_slots = minibatch_size - difficulty_slots - diversity_slots
        
        selected_indices = set()
        
        # Strategy 1: Difficulty-aware sampling
        difficulty_selected = self._difficulty_aware_sampling(analyses, difficulty_slots)
        selected_indices.update(difficulty_selected)
        
        # Strategy 2: Diversity-based sampling (excluding already selected)
        remaining_analyses = [a for a in analyses if a.instance_id not in selected_indices]
        diversity_selected = self._diversity_based_sampling(remaining_analyses, diversity_slots)
        selected_indices.update(diversity_selected)
        
        # Strategy 3: Error-focused sampling (excluding already selected)
        remaining_analyses = [a for a in analyses if a.instance_id not in selected_indices]
        error_selected = self._error_focused_sampling(remaining_analyses, error_slots)
        selected_indices.update(error_selected)
        
        return list(selected_indices)
    
    def _aggressive_sampling(
        self,
        analyses: List[InstanceAnalysis],
        minibatch_size: int
    ) -> List[int]:
        """Aggressive sampling - focus on hard and error-prone instances."""
        
        # Weight hard and error-prone instances heavily
        scores = []
        for analysis in analyses:
            score = (
                0.4 * analysis.difficulty_score +
                0.4 * analysis.error_likelihood +
                0.1 * analysis.uncertainty_score +
                0.1 * analysis.diversity_score
            )
            scores.append((score, analysis.instance_id))
        
        # Select top scoring instances
        scores.sort(reverse=True)
        return [instance_id for _, instance_id in scores[:minibatch_size]]
    
    def _conservative_sampling(
        self,
        analyses: List[InstanceAnalysis],
        minibatch_size: int
    ) -> List[int]:
        """Conservative sampling - focus on medium difficulty and diverse instances."""
        
        scores = []
        for analysis in analyses:
            # Prefer medium difficulty (around 0.5)
            difficulty_preference = 1.0 - abs(analysis.difficulty_score - 0.5) * 2
            
            score = (
                0.3 * difficulty_preference +
                0.4 * analysis.diversity_score +
                0.2 * analysis.uncertainty_score +
                0.1 * (1.0 - analysis.error_likelihood)  # Avoid error-prone
            )
            scores.append((score, analysis.instance_id))
        
        scores.sort(reverse=True)
        return [instance_id for _, instance_id in scores[:minibatch_size]]
    
    def _difficulty_aware_sampling(
        self,
        analyses: List[InstanceAnalysis],
        num_samples: int
    ) -> List[int]:
        """Sample based on predicted difficulty for current candidate."""
        
        # Target medium-difficulty instances (most informative)
        target_difficulty = 0.5
        
        # Calculate preference for each instance
        preferences = []
        for analysis in analyses:
            # Preference peaks at target difficulty
            preference = 1.0 - abs(analysis.difficulty_score - target_difficulty)
            preferences.append((preference, analysis.instance_id))
        
        # Sample based on preferences (probabilistic)
        preferences.sort(reverse=True)
        
        # Take top candidates but add some randomness
        candidate_pool = preferences[:min(num_samples * 2, len(preferences))]
        weights = [pref for pref, _ in candidate_pool]
        
        if sum(weights) == 0:
            # All weights are zero, use uniform sampling
            selected_indices = [instance_id for _, instance_id in candidate_pool[:num_samples]]
        else:
            # Weighted sampling
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            selected_indices = np.random.choice(
                [instance_id for _, instance_id in candidate_pool],
                size=min(num_samples, len(candidate_pool)),
                replace=False,
                p=weights
            ).tolist()
        
        return selected_indices
    
    def _diversity_based_sampling(
        self,
        analyses: List[InstanceAnalysis],
        num_samples: int
    ) -> List[int]:
        """Sample to maximize diversity in the minibatch."""
        
        if not analyses:
            return []
        
        # Greedy diversity sampling
        selected = []
        remaining = analyses.copy()
        
        # Start with most diverse instance
        most_diverse = max(remaining, key=lambda a: a.diversity_score)
        selected.append(most_diverse.instance_id)  
        remaining.remove(most_diverse)
        
        # Iteratively add most diverse remaining instances
        while len(selected) < num_samples and remaining:
            # Recalculate diversity scores relative to selected instances
            best_candidate = None
            best_score = -1
            
            for candidate in remaining:
                # Simple diversity score - in practice would calculate actual distance
                diversity_score = candidate.diversity_score
                
                if diversity_score > best_score:
                    best_score = diversity_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate.instance_id)
                remaining.remove(best_candidate)
            else:
                break
        
        return selected
    
    def _error_focused_sampling(
        self,
        analyses: List[InstanceAnalysis],
        num_samples: int
    ) -> List[int]:
        """Sample instances likely to reveal errors or weaknesses."""
        
        # Sort by error likelihood
        error_ranked = sorted(analyses, key=lambda a: a.error_likelihood, reverse=True)
        
        # Select top error-prone instances
        return [a.instance_id for a in error_ranked[:num_samples]]
    
    def _calculate_uncertainty_scores(
        self,
        training_dataset: List[Dict[str, Any]],
        candidate: Any
    ) -> Dict[int, float]:
        """Calculate uncertainty scores for instances."""
        
        uncertainty_scores = {}
        
        # If candidate has prediction confidence, use that
        if hasattr(candidate, 'prediction_confidence'):
            for i, confidence in enumerate(candidate.prediction_confidence):
                uncertainty_scores[i] = 1.0 - confidence
        else:
            # Default uncertainty based on performance variance
            if hasattr(candidate, 'scores') and candidate.scores:
                scores = list(candidate.scores.values())
                if len(scores) > 1:
                    score_variance = np.var(scores)
                    # High variance = high uncertainty
                    default_uncertainty = min(1.0, score_variance * 2)
                else:
                    default_uncertainty = 0.5
            else:
                default_uncertainty = 0.5
            
            uncertainty_scores = {i: default_uncertainty for i in range(len(training_dataset))}
        
        return uncertainty_scores
    
    def update_sampling_feedback(
        self,
        instance_ids: List[int],
        results: List[Dict[str, Any]]
    ) -> None:
        """Update sampling strategy based on results."""
        
        # Record errors for future sampling
        for i, result in enumerate(results):
            if i < len(instance_ids):
                instance_id = instance_ids[i]
                if result.get('error', False) or result.get('score', 1.0) < 0.3:
                    candidate_id = result.get('candidate_id', 'unknown')
                    self.error_sampler.record_error(instance_id, candidate_id)
    
    def get_sampling_statistics(self) -> Dict[str, Any]:
        """Get statistics about sampling performance."""
        
        if not self.sampling_history:
            return {"message": "No sampling history available"}
        
        return {
            "total_minibatches": len(self.sampling_history),
            "avg_minibatch_size": np.mean([len(batch) for batch in self.sampling_history]),
            "unique_instances_sampled": len(set().union(*self.sampling_history)),
            "instance_sampling_frequency": self._calculate_sampling_frequency()
        }
    
    def _calculate_sampling_frequency(self) -> Dict[int, int]:
        """Calculate how often each instance has been sampled."""
        
        frequency = defaultdict(int)
        for batch in self.sampling_history:
            for instance_id in batch:
                frequency[instance_id] += 1
        
        return dict(frequency) 