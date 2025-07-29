"""Pareto frontier management for GEPA optimization."""

import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import numpy as np


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
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.candidates: List[Candidate] = []
    
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