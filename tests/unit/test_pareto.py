"""Unit tests for Pareto frontier management."""

import pytest
from typing import List

from gepa.core.pareto import ParetoFrontier, Candidate


class TestParetoFrontier:
    """Test cases for ParetoFrontier class."""
    
    def test_empty_frontier(self):
        """Test empty frontier initialization."""
        frontier = ParetoFrontier(max_size=5)
        
        assert frontier.size() == 0
        assert frontier.is_empty()
        assert frontier.sample_candidate() is None
    
    def test_add_single_candidate(self, sample_candidate: Candidate):
        """Test adding a single candidate."""
        frontier = ParetoFrontier(max_size=5)
        
        added = frontier.add_candidate(sample_candidate)
        
        assert added
        assert frontier.size() == 1
        assert not frontier.is_empty()
        assert sample_candidate in frontier.candidates
    
    def test_add_dominated_candidate(self, sample_candidates: List[Candidate]):
        """Test that dominated candidates are not added."""
        frontier = ParetoFrontier(max_size=5)
        
        # Add best candidate first
        best_candidate = sample_candidates[0]  # highest scores
        frontier.add_candidate(best_candidate)
        
        # Try to add a dominated candidate
        dominated_candidate = sample_candidates[3]  # lower scores, higher cost
        added = frontier.add_candidate(dominated_candidate)
        
        assert not added
        assert frontier.size() == 1
        assert dominated_candidate not in frontier.candidates
    
    def test_add_dominating_candidate(self, sample_candidates: List[Candidate]):
        """Test that dominating candidates replace dominated ones."""
        frontier = ParetoFrontier(max_size=5)
        
        # Add weaker candidate first
        weak_candidate = sample_candidates[3]
        frontier.add_candidate(weak_candidate)
        
        # Add stronger candidate
        strong_candidate = sample_candidates[0]
        added = frontier.add_candidate(strong_candidate)
        
        assert added
        assert frontier.size() == 1
        assert weak_candidate not in frontier.candidates
        assert strong_candidate in frontier.candidates
    
    def test_pareto_optimality(self, sample_candidates: List[Candidate]):
        """Test that frontier maintains Pareto optimality."""
        frontier = ParetoFrontier(max_size=10)
        
        # Add all candidates
        for candidate in sample_candidates:
            frontier.add_candidate(candidate)
        
        # Verify no candidate dominates another in the frontier
        for i, candidate1 in enumerate(frontier.candidates):
            for j, candidate2 in enumerate(frontier.candidates):
                if i != j:
                    assert not candidate1.dominates(candidate2)
    
    def test_max_size_enforcement(self, sample_candidates: List[Candidate]):
        """Test that frontier enforces maximum size."""
        max_size = 3
        frontier = ParetoFrontier(max_size=max_size)
        
        # Add more candidates than max size
        for candidate in sample_candidates:
            frontier.add_candidate(candidate)
        
        assert frontier.size() <= max_size
    
    def test_sample_candidate(self, sample_candidates: List[Candidate]):
        """Test candidate sampling from frontier."""
        frontier = ParetoFrontier(max_size=5)
        
        # Add candidates
        for candidate in sample_candidates[:3]:
            frontier.add_candidate(candidate)
        
        # Test sampling
        sampled = frontier.sample_candidate()
        assert sampled is not None
        assert sampled in frontier.candidates
        
        # Test sampling with different beta values
        exploitation_sample = frontier.sample_candidate(beta=1.0)
        exploration_sample = frontier.sample_candidate(beta=0.0)
        
        assert exploitation_sample in frontier.candidates
        assert exploration_sample in frontier.candidates
    
    def test_get_diverse_sample(self, sample_candidates: List[Candidate]):
        """Test getting diverse samples from frontier."""
        frontier = ParetoFrontier(max_size=5)
        
        for candidate in sample_candidates:
            frontier.add_candidate(candidate)
        
        diverse_sample = frontier.get_diverse_sample(2)
        
        assert len(diverse_sample) <= 2
        assert len(diverse_sample) <= frontier.size()
        
        for candidate in diverse_sample:
            assert candidate in frontier.candidates
    
    def test_get_best_candidates(self, sample_candidates: List[Candidate]):
        """Test getting best candidates by specific metric."""
        frontier = ParetoFrontier(max_size=5)
        
        for candidate in sample_candidates:
            frontier.add_candidate(candidate)
        
        best_by_f1 = frontier.get_best_candidates(2, "f1_score")
        
        assert len(best_by_f1) <= 2
        assert len(best_by_f1) <= frontier.size()
        
        # Verify ordering
        if len(best_by_f1) > 1:
            for i in range(len(best_by_f1) - 1):
                assert (best_by_f1[i].scores.get("f1_score", 0) >= 
                       best_by_f1[i + 1].scores.get("f1_score", 0))
    
    def test_frontier_statistics(self, sample_candidates: List[Candidate]):
        """Test frontier statistics calculation."""
        frontier = ParetoFrontier(max_size=5)
        
        for candidate in sample_candidates:
            frontier.add_candidate(candidate)
        
        stats = frontier.get_statistics()
        
        assert "size" in stats
        assert "metrics" in stats
        assert "cost" in stats
        assert "tokens" in stats
        
        assert stats["size"] == frontier.size()
        assert stats["size"] > 0
        
        # Check metric statistics
        for metric_name in ["exact_match", "f1_score"]:
            if metric_name in stats["metrics"]:
                metric_stats = stats["metrics"][metric_name]
                assert "min" in metric_stats
                assert "max" in metric_stats
                assert "mean" in metric_stats
                assert "std" in metric_stats
    
    def test_clear_frontier(self, sample_candidates: List[Candidate]):
        """Test clearing the frontier."""
        frontier = ParetoFrontier(max_size=5)
        
        for candidate in sample_candidates:
            frontier.add_candidate(candidate)
        
        assert frontier.size() > 0
        
        frontier.clear()
        
        assert frontier.size() == 0
        assert frontier.is_empty()
        assert len(frontier.candidates) == 0


class TestCandidate:
    """Test cases for Candidate class."""
    
    def test_candidate_creation(self, sample_system):
        """Test candidate creation."""
        candidate = Candidate(
            id="test-candidate",
            system=sample_system,
            scores={"metric1": 0.8, "metric2": 0.7},
            cost=0.05,
            tokens_used=100
        )
        
        assert candidate.id == "test-candidate"
        assert candidate.system == sample_system
        assert candidate.scores == {"metric1": 0.8, "metric2": 0.7}
        assert candidate.cost == 0.05
        assert candidate.tokens_used == 100
    
    def test_candidate_dominance(self, sample_system):
        """Test candidate dominance relationships."""
        # Create candidates with different performance profiles
        candidate1 = Candidate(
            id="c1",
            system=sample_system,
            scores={"metric1": 0.9, "metric2": 0.8},
            cost=0.05,
            tokens_used=100
        )
        
        candidate2 = Candidate(
            id="c2", 
            system=sample_system,
            scores={"metric1": 0.7, "metric2": 0.6},
            cost=0.08,
            tokens_used=120
        )
        
        candidate3 = Candidate(
            id="c3",
            system=sample_system,
            scores={"metric1": 0.9, "metric2": 0.7},
            cost=0.05,
            tokens_used=100
        )
        
        # Test dominance
        assert candidate1.dominates(candidate2)  # Better in all metrics and cost
        assert not candidate2.dominates(candidate1)  # Worse in all metrics
        assert not candidate1.dominates(candidate3)  # Mixed performance
        assert not candidate3.dominates(candidate1)  # Mixed performance
    
    def test_candidate_equality(self, sample_system):
        """Test candidate equality based on ID."""
        candidate1 = Candidate(
            id="same-id",
            system=sample_system,
            scores={"metric1": 0.8},
            cost=0.05,
            tokens_used=100
        )
        
        candidate2 = Candidate(
            id="same-id",
            system=sample_system,
            scores={"metric1": 0.9},  # Different scores
            cost=0.06,  # Different cost
            tokens_used=120  # Different tokens
        )
        
        candidate3 = Candidate(
            id="different-id",
            system=sample_system,
            scores={"metric1": 0.8},
            cost=0.05,
            tokens_used=100
        )
        
        assert candidate1 == candidate2  # Same ID
        assert candidate1 != candidate3  # Different ID
        
        # Test hashing
        candidate_set = {candidate1, candidate2, candidate3}
        assert len(candidate_set) == 2  # Only 2 unique IDs