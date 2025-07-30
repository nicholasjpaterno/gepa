"""Pareto frontier management for GEPA optimization."""

import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import numpy as np

# Import advanced algorithms when available
try:
    from ..algorithms.advanced.score_prediction import InstanceScorePredictor
    from ..algorithms.advanced.adaptive_comparison import AdaptiveScoreComparator, ComparisonContext
    ADVANCED_ALGORITHMS_AVAILABLE = True
except ImportError:
    ADVANCED_ALGORITHMS_AVAILABLE = False


@dataclass
class Candidate:
    """A candidate solution in the optimization."""
    id: str
    system: Any  # CompoundAISystem
    scores: Dict[str, float]  # Metric name -> score
    cost: float
    tokens_used: int
    parent_id: Optional[str] = None
    generation: int = 0
    
    def dominates(self, other: "Candidate") -> bool:
        """Check if this candidate dominates another (Pareto dominance)."""
        # A candidate dominates another if it's better or equal in all objectives
        # and strictly better in at least one objective
        better_in_any = False
        
        for metric_name in self.scores:
            if metric_name not in other.scores:
                continue
                
            self_score = self.scores[metric_name]
            other_score = other.scores[metric_name]
            
            # Higher scores are better
            if self_score < other_score:
                return False  # Worse in this metric
            elif self_score > other_score:
                better_in_any = True
        
        # Also consider cost (lower is better)
        if self.cost > other.cost:
            return False
        elif self.cost < other.cost:
            better_in_any = True
            
        return better_in_any
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Candidate):
            return False
        return self.id == other.id


class ParetoFrontier:
    """Manages the Pareto frontier of candidate solutions."""
    
    def __init__(self, max_size: int = 10, config: Optional[Any] = None):
        self.max_size = max_size
        self.candidates: List[Candidate] = []
        self.config = config
        
        # Initialize advanced algorithms if available and enabled
        self.score_predictor = None
        self.score_comparator = None
        
        if ADVANCED_ALGORITHMS_AVAILABLE and config and hasattr(config, 'advanced'):
            if config.advanced.enable_score_prediction:
                self.score_predictor = InstanceScorePredictor()
            
            if config.advanced.enable_adaptive_comparison:
                confidence_threshold = getattr(config.advanced, 'comparison_confidence_threshold', 0.95)
                self.score_comparator = AdaptiveScoreComparator(confidence_threshold)
    
    def add_candidate(self, candidate: Candidate) -> bool:
        """
        Add a candidate to the Pareto frontier.
        Returns True if candidate was added, False if dominated.
        """
        # Check if candidate is dominated by any existing candidate
        for existing in self.candidates:
            if existing.dominates(candidate):
                return False  # Candidate is dominated
        
        # Remove any existing candidates dominated by the new candidate
        self.candidates = [
            c for c in self.candidates 
            if not candidate.dominates(c)
        ]
        
        # Add the new candidate
        self.candidates.append(candidate)
        
        # If we exceed max size, remove least diverse candidate
        if len(self.candidates) > self.max_size:
            self._maintain_diversity()
        
        return True
    
    def _maintain_diversity(self) -> None:
        """Remove candidates to maintain diversity when frontier is too large."""
        if len(self.candidates) <= self.max_size:
            return
        
        # Calculate crowding distance for each candidate
        crowding_distances = self._calculate_crowding_distances()
        
        # Sort by crowding distance (descending) and keep most diverse
        candidates_with_distance = list(zip(self.candidates, crowding_distances))
        candidates_with_distance.sort(key=lambda x: x[1], reverse=True)
        
        self.candidates = [
            candidate for candidate, _ in candidates_with_distance[:self.max_size]
        ]
    
    def _calculate_crowding_distances(self) -> List[float]:
        """Calculate crowding distance for diversity maintenance."""
        if len(self.candidates) <= 2:
            return [float('inf')] * len(self.candidates)
        
        distances = [0.0] * len(self.candidates)
        
        # Get all metrics
        all_metrics = set()
        for candidate in self.candidates:
            all_metrics.update(candidate.scores.keys())
        all_metrics.add('cost')  # Include cost as objective
        
        # Calculate crowding distance for each metric
        for metric in all_metrics:
            # Get values for this metric
            if metric == 'cost':
                values = [(i, candidate.cost) for i, candidate in enumerate(self.candidates)]
            else:
                values = [
                    (i, candidate.scores.get(metric, 0.0)) 
                    for i, candidate in enumerate(self.candidates)
                ]
            
            # Sort by metric value
            values.sort(key=lambda x: x[1])
            
            # Set boundary points to infinity
            if len(values) > 2:
                distances[values[0][0]] = float('inf')
                distances[values[-1][0]] = float('inf')
                
                # Calculate normalized distance for interior points
                metric_range = values[-1][1] - values[0][1]
                if metric_range > 0:
                    for i in range(1, len(values) - 1):
                        curr_idx = values[i][0]
                        distance = (values[i+1][1] - values[i-1][1]) / metric_range
                        distances[curr_idx] += distance
        
        return distances
    
    def sample_candidate_algorithm2(
        self, 
        training_dataset: List[Dict[str, Any]], 
        scores_matrix: Optional[Dict[str, Dict[int, float]]] = None
    ) -> Optional[Candidate]:
        """
        Algorithm 2: Pareto-based candidate selection from the paper.
        
        Implements the exact algorithm from the research paper:
        1. Build instance-wise Pareto sets
        2. Find unique candidates in union of Pareto sets
        3. Remove dominated candidates
        4. Sample proportionally based on frequency
        
        Args:
            training_dataset: The training dataset instances
            scores_matrix: Optional precomputed scores matrix
            
        Returns:
            Selected candidate or None if frontier is empty
        """
        if not self.candidates or not training_dataset:
            return None
            
        P = self.candidates  # Candidate pool
        n_instances = len(training_dataset)
        
        # Step 1-2: Build instance-wise Pareto sets
        pareto_sets_per_instance = []  # P*[i] for each instance i
        
        for i in range(n_instances):
            # Find max score for instance i across all candidates
            max_score_i = float('-inf')
            for candidate in P:
                if scores_matrix and candidate.id in scores_matrix:
                    score_i = scores_matrix[candidate.id].get(i, 0.0)
                else:
                    # Use advanced score prediction if available, otherwise fallback
                    if self.score_predictor:
                        training_data = {'instances': training_dataset}
                        score_i = self.score_predictor.predict_instance_score(candidate, i, training_data)
                    else:
                        # Fallback: use average score as proxy
                        score_i = sum(candidate.scores.values()) / max(len(candidate.scores), 1)
                max_score_i = max(max_score_i, score_i)
            
            # Find all candidates that achieve max score on instance i
            pareto_set_i = []
            for candidate in P:
                if scores_matrix and candidate.id in scores_matrix:
                    score_i = scores_matrix[candidate.id].get(i, 0.0)
                else:
                    # Use advanced score prediction if available, otherwise fallback  
                    if self.score_predictor:
                        training_data = {'instances': training_dataset}
                        score_i = self.score_predictor.predict_instance_score(candidate, i, training_data)
                    else:
                        score_i = sum(candidate.scores.values()) / max(len(candidate.scores), 1)
                
                # Use advanced score comparison if available, otherwise fixed epsilon
                if self.score_comparator:
                    context = ComparisonContext(score_variance=0.01, sample_size=len(P))
                    comparison_result = self.score_comparator.scores_equivalent(score_i, max_score_i, context)
                    if comparison_result.are_equivalent:
                        pareto_set_i.append(candidate)
                else:
                    if abs(score_i - max_score_i) < 1e-6:  # Handle floating point precision
                        pareto_set_i.append(candidate)
            
            pareto_sets_per_instance.append(pareto_set_i)
        
        # Step 3: Get unique candidates from union of all Pareto sets
        C = set()
        for pareto_set in pareto_sets_per_instance:
            C.update(pareto_set)
        C = list(C)
        
        if not C:
            return None
        
        # Step 4-5: Remove dominated candidates
        D = set()  # Dominated candidates to remove
        
        for candidate in C:
            if candidate in D:
                continue
                
            for other in C:
                if other != candidate and other not in D:
                    if other.dominates(candidate):
                        D.add(candidate)
                        break
        
        # Step 6: Remove dominated candidates from each Pareto set
        filtered_pareto_sets = []
        for pareto_set in pareto_sets_per_instance:
            filtered_set = [c for c in pareto_set if c not in D]
            filtered_pareto_sets.append(filtered_set)
        
        # Step 7: Calculate frequency f[Φ] for each candidate
        frequency = {}
        remaining_candidates = [c for c in C if c not in D]
        
        for candidate in remaining_candidates:
            count = 0
            for filtered_set in filtered_pareto_sets:
                if candidate in filtered_set:
                    count += 1
            frequency[candidate] = count
        
        if not frequency:
            return None
        
        # Step 8: Sample candidate proportionally to frequency
        candidates_list = list(frequency.keys())
        frequencies = list(frequency.values())
        total_frequency = sum(frequencies)
        
        if total_frequency == 0:
            return random.choice(candidates_list)
        
        # Weighted random selection
        probabilities = [f / total_frequency for f in frequencies]
        
        # Use numpy for weighted selection
        import numpy as np
        selected_idx = np.random.choice(len(candidates_list), p=probabilities)
        
        return candidates_list[selected_idx]
    
    def sample_candidate(self, beta: float = 0.5) -> Optional[Candidate]:
        """
        Sample a candidate from the Pareto frontier.
        
        Args:
            beta: Balance between exploration (0) and exploitation (1)
        """
        if not self.candidates:
            return None
        
        if random.random() < beta:
            # Exploitation: select best performing candidate
            best_candidate = max(
                self.candidates,
                key=lambda c: sum(c.scores.values()) / max(len(c.scores), 1)
            )
            return best_candidate
        else:
            # Exploration: sample uniformly
            return random.choice(self.candidates)
    
    def get_diverse_sample(self, k: int) -> List[Candidate]:
        """Get k diverse candidates from the frontier."""
        if k >= len(self.candidates):
            return self.candidates.copy()
        
        # Use crowding distance to select diverse candidates
        crowding_distances = self._calculate_crowding_distances()
        candidates_with_distance = list(zip(self.candidates, crowding_distances))
        candidates_with_distance.sort(key=lambda x: x[1], reverse=True)
        
        return [candidate for candidate, _ in candidates_with_distance[:k]]
    
    def get_best_candidates(self, k: int, metric: str) -> List[Candidate]:
        """Get top k candidates by a specific metric."""
        candidates_with_score = [
            (candidate, candidate.scores.get(metric, 0.0))
            for candidate in self.candidates
        ]
        candidates_with_score.sort(key=lambda x: x[1], reverse=True)
        
        return [candidate for candidate, _ in candidates_with_score[:k]]
    
    def size(self) -> int:
        """Get number of candidates in frontier."""
        return len(self.candidates)
    
    def is_empty(self) -> bool:
        """Check if frontier is empty."""
        return len(self.candidates) == 0
    
    def clear(self) -> None:
        """Clear all candidates."""
        self.candidates.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the Pareto frontier."""
        if not self.candidates:
            return {"size": 0, "metrics": {}}
        
        # Collect all metrics
        all_metrics = set()
        for candidate in self.candidates:
            all_metrics.update(candidate.scores.keys())
        
        stats = {
            "size": len(self.candidates),
            "metrics": {},
            "cost": {
                "min": min(c.cost for c in self.candidates),
                "max": max(c.cost for c in self.candidates),
                "mean": np.mean([c.cost for c in self.candidates]),
            },
            "tokens": {
                "min": min(c.tokens_used for c in self.candidates),
                "max": max(c.tokens_used for c in self.candidates),
                "mean": np.mean([c.tokens_used for c in self.candidates]),
            }
        }
        
        for metric in all_metrics:
            scores = [c.scores.get(metric, 0.0) for c in self.candidates]
            stats["metrics"][metric] = {
                "min": min(scores),
                "max": max(scores),
                "mean": np.mean(scores),
                "std": np.std(scores),
            }
        
        return stats