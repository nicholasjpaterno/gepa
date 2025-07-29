"""Base evaluation interfaces and abstractions."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class EvaluationResult:
    """Result of evaluation."""
    scores: Dict[str, float]  # Metric name -> score
    cost: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class Metric(ABC):
    """Abstract base class for evaluation metrics."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def compute(
        self, 
        predictions: List[Any], 
        references: List[Any]
    ) -> float:
        """
        Compute the metric score.
        
        Args:
            predictions: List of predicted outputs
            references: List of reference/ground truth outputs
        
        Returns:
            Metric score (higher is better)
        """
        pass
    
    def batch_compute(
        self,
        predictions: List[Any],
        references: List[Any]
    ) -> List[float]:
        """
        Compute metric for each prediction-reference pair.
        
        Returns:
            List of individual scores
        """
        return [
            self.compute([pred], [ref]) 
            for pred, ref in zip(predictions, references)
        ]


class Evaluator(ABC):
    """Abstract base class for evaluators."""
    
    def __init__(self, metrics: List[Metric]):
        self.metrics = {metric.name: metric for metric in metrics}
    
    @abstractmethod
    async def evaluate_batch(
        self,
        predictions: List[Any],
        references: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """
        Evaluate a batch of predictions.
        
        Args:
            predictions: List of system outputs to evaluate
            references: List of reference data with expected outputs
        
        Returns:
            EvaluationResult with scores for all metrics
        """
        pass
    
    async def evaluate_single(
        self,
        prediction: Any,
        reference: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate a single prediction."""
        return await self.evaluate_batch([prediction], [reference])
    
    def add_metric(self, metric: Metric) -> None:
        """Add a metric to the evaluator."""
        self.metrics[metric.name] = metric
    
    def remove_metric(self, metric_name: str) -> None:
        """Remove a metric from the evaluator."""
        if metric_name in self.metrics:
            del self.metrics[metric_name]
    
    def get_metric_names(self) -> List[str]:
        """Get names of all metrics."""
        return list(self.metrics.keys())


class SimpleEvaluator(Evaluator):
    """Simple evaluator that computes metrics directly."""
    
    async def evaluate_batch(
        self,
        predictions: List[Any],
        references: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """Evaluate predictions using configured metrics."""
        if len(predictions) != len(references):
            raise ValueError("Number of predictions must match references")
        
        scores = {}
        
        for metric_name, metric in self.metrics.items():
            try:
                # Extract expected values from references
                expected_values = []
                for ref in references:
                    if "expected" in ref:
                        expected_values.append(ref["expected"])
                    else:
                        # Try to infer expected value from reference
                        expected_values.append(ref)
                
                score = metric.compute(predictions, expected_values)
                scores[metric_name] = score
                
            except Exception as e:
                # If metric computation fails, set score to 0
                scores[metric_name] = 0.0
        
        return EvaluationResult(scores=scores)


class LLMEvaluator(Evaluator):
    """Evaluator that uses an LLM for evaluation."""
    
    def __init__(
        self,
        metrics: List[Metric],
        llm_client: Any,  # InferenceClient
        evaluation_prompt_template: str
    ):
        super().__init__(metrics)
        self.llm_client = llm_client
        self.evaluation_prompt_template = evaluation_prompt_template
    
    async def evaluate_batch(
        self,
        predictions: List[Any],
        references: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """Evaluate using LLM-based evaluation."""
        from ..inference.base import InferenceRequest
        
        scores = {}
        total_cost = 0.0
        
        # First compute traditional metrics
        traditional_result = await super().evaluate_batch(predictions, references)
        scores.update(traditional_result.scores)
        
        # Then add LLM-based evaluation
        llm_scores = []
        
        for pred, ref in zip(predictions, references):
            # Create evaluation prompt
            evaluation_prompt = self.evaluation_prompt_template.format(
                prediction=pred,
                reference=ref.get("expected", ref),
                context=ref
            )
            
            request = InferenceRequest(
                prompt=evaluation_prompt,
                max_tokens=100,
                temperature=0.1  # Low temperature for consistent evaluation
            )
            
            try:
                response = await self.llm_client.generate(request)
                
                # Parse score from response (assumes response contains numeric score)
                score_text = response.text.strip()
                score = self._parse_score_from_response(score_text)
                llm_scores.append(score)
                
                if response.cost:
                    total_cost += response.cost
                    
            except Exception:
                llm_scores.append(0.0)  # Default score on error
        
        # Average LLM scores
        if llm_scores:
            scores["llm_score"] = sum(llm_scores) / len(llm_scores)
        
        return EvaluationResult(
            scores=scores,
            cost=total_cost
        )
    
    def _parse_score_from_response(self, response: str) -> float:
        """Parse numeric score from LLM response."""
        import re
        
        # Look for patterns like "Score: 8.5" or "8/10" or just "0.85"
        patterns = [
            r'score:\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*/\s*10',
            r'(\d+(?:\.\d+)?)\s*/\s*100',
            r'^(\d+(?:\.\d+)?)$'
        ]
        
        response_lower = response.lower()
        
        for pattern in patterns:
            match = re.search(pattern, response_lower)
            if match:
                score = float(match.group(1))
                
                # Normalize to 0-1 range
                if "/10" in response_lower:
                    score = score / 10.0
                elif "/100" in response_lower:
                    score = score / 100.0
                elif score > 1.0 and score <= 10.0:
                    score = score / 10.0
                elif score > 10.0:
                    score = score / 100.0
                
                return max(0.0, min(1.0, score))
        
        # Default to 0.5 if no score found
        return 0.5