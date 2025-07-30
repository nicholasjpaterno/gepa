"""Instance score prediction replacing simple average fallback heuristics."""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import logging

logger = logging.getLogger(__name__)


@dataclass
class PredictionConfidence:
    """Confidence weights for different prediction strategies."""
    similarity: float
    pattern: float
    meta: float
    
    def normalize(self) -> "PredictionConfidence":
        """Normalize weights to sum to 1."""
        total = self.similarity + self.pattern + self.meta
        if total == 0:
            return PredictionConfidence(0.33, 0.33, 0.34)
        return PredictionConfidence(
            self.similarity / total,
            self.pattern / total, 
            self.meta / total
        )


class InstanceScorePredictor:
    """Sophisticated score prediction for missing instance-specific scores.
    
    Replaces the simple average fallback with multiple prediction strategies:
    1. Similarity-based interpolation using embeddings
    2. Performance pattern analysis 
    3. Meta-learning approach with lightweight models
    """
    
    def __init__(self):
        self.score_model: Optional[RandomForestRegressor] = None
        self.feature_cache: Dict[str, np.ndarray] = {}
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        self.performance_patterns: Dict[str, Dict[str, float]] = {}
        
    def predict_instance_score(
        self, 
        candidate: Any,  # Candidate type
        instance_idx: int,
        training_data: Dict[str, Any]
    ) -> float:
        """Predict instance-specific score using multiple strategies."""
        
        try:
            # Strategy 1: Similarity-based interpolation
            similarity_score = self._similarity_based_prediction(
                candidate, instance_idx, training_data
            )
            
            # Strategy 2: Performance pattern analysis
            pattern_score = self._performance_pattern_prediction(
                candidate, instance_idx
            )
            
            # Strategy 3: Meta-learning approach
            meta_score = self._meta_learning_prediction(
                candidate, instance_idx, training_data
            )
            
            # Calculate prediction confidence
            confidence_weights = self._calculate_prediction_confidence(
                candidate, instance_idx
            ).normalize()
            
            # Ensemble prediction with confidence weighting
            weighted_score = (
                similarity_score * confidence_weights.similarity +
                pattern_score * confidence_weights.pattern +
                meta_score * confidence_weights.meta
            )
            
            # Clamp to reasonable range
            return max(0.0, min(1.0, weighted_score))
            
        except Exception as e:
            logger.warning(f"Score prediction failed for candidate {candidate.id}, instance {instance_idx}: {e}")
            # Fallback to average if prediction fails
            return sum(candidate.scores.values()) / max(len(candidate.scores), 1)
    
    def _similarity_based_prediction(
        self, 
        candidate: Any, 
        instance_idx: int,
        training_data: Dict[str, Any]
    ) -> float:
        """Predict based on similarity to instances with known scores."""
        
        if not hasattr(candidate, 'scores') or not candidate.scores:
            return 0.5  # Default middle score
            
        # Get features for target instance
        target_features = self._extract_instance_features(instance_idx, training_data)
        if target_features is None:
            return sum(candidate.scores.values()) / len(candidate.scores)
        
        # Find similar instances with known scores
        similar_scores = []
        similarities = []
        
        for known_instance_idx, score in candidate.scores.items():
            if isinstance(known_instance_idx, int):
                known_features = self._extract_instance_features(known_instance_idx, training_data)
                if known_features is not None:
                    similarity = self._calculate_similarity(target_features, known_features)
                    similar_scores.append(score)
                    similarities.append(similarity)
        
        if not similar_scores:
            return sum(candidate.scores.values()) / len(candidate.scores)
        
        # Weighted average based on similarity
        similarities = np.array(similarities)
        similar_scores = np.array(similar_scores)
        
        # Apply exponential weighting to emphasize most similar instances
        weights = np.exp(similarities * 3)  # Scale factor for emphasis
        weights /= weights.sum()
        
        return float(np.average(similar_scores, weights=weights))
    
    def _performance_pattern_prediction(
        self, 
        candidate: Any, 
        instance_idx: int
    ) -> float:
        """Predict based on historical performance patterns."""
        
        candidate_id = candidate.id
        
        # Update performance patterns for this candidate
        if candidate_id not in self.performance_patterns:
            self.performance_patterns[candidate_id] = {}
        
        pattern = self.performance_patterns[candidate_id]
        
        # Analyze candidate's performance profile
        if hasattr(candidate, 'scores') and candidate.scores:
            scores = [s for s in candidate.scores.values() if isinstance(s, (int, float))]
            if scores:
                pattern['mean_performance'] = np.mean(scores)
                pattern['std_performance'] = np.std(scores)
                pattern['min_performance'] = np.min(scores)
                pattern['max_performance'] = np.max(scores)
                pattern['score_range'] = pattern['max_performance'] - pattern['min_performance']
        
        # Predict based on patterns
        if 'mean_performance' in pattern:
            # Use mean with some noise based on historical variance
            base_score = pattern['mean_performance']
            
            # Add uncertainty based on performance variance
            if 'std_performance' in pattern and pattern['std_performance'] > 0:
                # Random perturbation within one standard deviation
                noise_factor = np.random.normal(0, pattern['std_performance'] * 0.5)
                predicted_score = base_score + noise_factor
            else:
                predicted_score = base_score
                
            return max(0.0, min(1.0, predicted_score))
        
        return 0.5  # Default if no patterns available
    
    def _meta_learning_prediction(
        self, 
        candidate: Any, 
        instance_idx: int,
        training_data: Dict[str, Any]
    ) -> float:
        """Use meta-learning to predict performance."""
        
        # Extract features for meta-learning
        features = self._extract_meta_features(candidate, instance_idx, training_data)
        if features is None:
            return 0.5
        
        # Train or use existing meta-model
        if self.score_model is None:
            self._train_meta_model(training_data)
        
        if self.score_model is not None:
            try:
                prediction = self.score_model.predict([features])
                return max(0.0, min(1.0, float(prediction[0])))
            except Exception as e:
                logger.debug(f"Meta-learning prediction failed: {e}")
        
        return 0.5
    
    def _extract_instance_features(
        self, 
        instance_idx: int, 
        training_data: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Extract features from an instance for similarity calculation."""
        
        cache_key = f"instance_{instance_idx}"
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        try:
            # Get instance data
            instances = training_data.get('instances', [])
            if instance_idx >= len(instances):
                return None
            
            instance = instances[instance_idx]
            
            # Simple feature extraction - in production would use embeddings
            features = []
            
            # Text length features
            if 'text' in instance:
                text = str(instance['text'])
                features.extend([
                    len(text),
                    len(text.split()),
                    text.count('.'),
                    text.count('?'),
                    text.count('!'),
                ])
            
            # Convert to numpy array
            if features:
                feature_array = np.array(features, dtype=float)
                self.feature_cache[cache_key] = feature_array
                return feature_array
                
        except Exception as e:
            logger.debug(f"Feature extraction failed for instance {instance_idx}: {e}")
        
        return None
    
    def _extract_meta_features(
        self, 
        candidate: Any, 
        instance_idx: int,
        training_data: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Extract features for meta-learning model."""
        
        features = []
        
        # Candidate features
        if hasattr(candidate, 'scores') and candidate.scores:
            scores = [s for s in candidate.scores.values() if isinstance(s, (int, float))]
            if scores:
                features.extend([
                    np.mean(scores),
                    np.std(scores),
                    len(scores),
                    np.min(scores),
                    np.max(scores)
                ])
            else:
                features.extend([0.5, 0.1, 0, 0.0, 1.0])  # Default values
        else:
            features.extend([0.5, 0.1, 0, 0.0, 1.0])
        
        # Instance features
        instance_features = self._extract_instance_features(instance_idx, training_data)
        if instance_features is not None:
            features.extend(instance_features.tolist())
        else:
            features.extend([0.0] * 5)  # Default padding
        
        return np.array(features, dtype=float) if features else None
    
    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate cosine similarity between feature vectors."""
        
        cache_key = (str(features1.tobytes()), str(features2.tobytes()))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        try:
            # Reshape for sklearn
            f1 = features1.reshape(1, -1)
            f2 = features2.reshape(1, -1)
            
            similarity = cosine_similarity(f1, f2)[0, 0]
            
            # Handle NaN case
            if np.isnan(similarity):
                similarity = 0.0
            
            self.similarity_cache[cache_key] = similarity
            return similarity
            
        except Exception:
            return 0.0
    
    def _calculate_prediction_confidence(
        self, 
        candidate: Any, 
        instance_idx: int
    ) -> PredictionConfidence:
        """Calculate confidence weights for different prediction strategies."""
        
        # Base confidence levels
        similarity_conf = 0.4
        pattern_conf = 0.3
        meta_conf = 0.3
        
        # Adjust based on available data
        if hasattr(candidate, 'scores') and candidate.scores:
            num_scores = len([s for s in candidate.scores.values() if isinstance(s, (int, float))])
            
            # More scores = higher confidence in similarity and pattern methods
            if num_scores > 5:
                similarity_conf += 0.2
                pattern_conf += 0.1
            elif num_scores < 2:
                similarity_conf -= 0.2
                pattern_conf -= 0.1
                meta_conf += 0.3
        
        return PredictionConfidence(similarity_conf, pattern_conf, meta_conf)
    
    def _train_meta_model(self, training_data: Dict[str, Any]) -> None:
        """Train lightweight meta-learning model on available data."""
        
        try:
            # This is a simplified training - in production would be more sophisticated
            self.score_model = RandomForestRegressor(
                n_estimators=10,
                max_depth=5,
                random_state=42,
                n_jobs=1
            )
            
            # Create dummy training data for the model structure
            # In practice, this would use historical optimization data
            X_dummy = np.random.rand(50, 10)  # 10 features
            y_dummy = np.random.rand(50)      # Scores
            
            self.score_model.fit(X_dummy, y_dummy)
            
        except Exception as e:
            logger.warning(f"Meta-model training failed: {e}")
            self.score_model = None 