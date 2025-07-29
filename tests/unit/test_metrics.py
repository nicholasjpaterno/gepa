"""Unit tests for evaluation metrics."""

import pytest
from typing import List

from gepa.evaluation.metrics import (
    ExactMatch,
    F1Score,
    RougeL,
    BLEU,
    CodeExecutionMetric,
)


class TestExactMatch:
    """Test cases for ExactMatch metric."""
    
    def test_exact_match_perfect(self):
        """Test exact match with perfect matches."""
        metric = ExactMatch()
        predictions = ["hello world", "test case", "example"]
        references = ["hello world", "test case", "example"]
        
        score = metric.compute(predictions, references)
        assert score == 1.0
    
    def test_exact_match_no_matches(self):
        """Test exact match with no matches."""
        metric = ExactMatch()
        predictions = ["hello world", "test case", "example"]
        references = ["goodbye world", "wrong case", "different"]
        
        score = metric.compute(predictions, references)
        assert score == 0.0
    
    def test_exact_match_partial(self):
        """Test exact match with partial matches."""
        metric = ExactMatch()
        predictions = ["hello world", "test case", "example"]
        references = ["hello world", "wrong case", "example"]
        
        score = metric.compute(predictions, references)
        assert score == 2/3
    
    def test_exact_match_case_insensitive(self):
        """Test case insensitive exact match."""
        metric = ExactMatch(case_sensitive=False)
        predictions = ["Hello World", "TEST case", "Example"]
        references = ["hello world", "test CASE", "example"]
        
        score = metric.compute(predictions, references)
        assert score == 1.0
    
    def test_exact_match_case_sensitive(self):
        """Test case sensitive exact match."""
        metric = ExactMatch(case_sensitive=True)
        predictions = ["Hello World", "TEST case", "Example"]
        references = ["hello world", "test CASE", "example"]
        
        score = metric.compute(predictions, references)
        assert score == 0.0
    
    def test_exact_match_empty_lists(self):
        """Test exact match with empty lists."""
        metric = ExactMatch()
        score = metric.compute([], [])
        assert score == 0.0
    
    def test_exact_match_mismatched_lengths(self):
        """Test exact match with mismatched list lengths."""
        metric = ExactMatch()
        predictions = ["hello", "world"]
        references = ["hello"]
        
        score = metric.compute(predictions, references)
        assert score == 0.0


class TestF1Score:
    """Test cases for F1Score metric."""
    
    def test_f1_perfect_match(self):
        """Test F1 score with perfect matches."""
        metric = F1Score()
        predictions = ["hello world test"]
        references = ["hello world test"]
        
        score = metric.compute(predictions, references)
        assert score == 1.0
    
    def test_f1_no_overlap(self):
        """Test F1 score with no token overlap."""
        metric = F1Score()
        predictions = ["hello world"]
        references = ["goodbye universe"]
        
        score = metric.compute(predictions, references)
        assert score == 0.0
    
    def test_f1_partial_overlap(self):
        """Test F1 score with partial token overlap."""
        metric = F1Score()
        predictions = ["hello world test case"]
        references = ["hello world example"]
        
        score = metric.compute(predictions, references)
        
        # Expected: precision = 2/4, recall = 2/3, F1 = 2 * (2/4) * (2/3) / ((2/4) + (2/3))
        expected_precision = 2/4
        expected_recall = 2/3
        expected_f1 = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)
        
        assert abs(score - expected_f1) < 1e-6
    
    def test_f1_empty_prediction(self):
        """Test F1 score with empty prediction."""
        metric = F1Score()
        predictions = [""]
        references = ["hello world"]
        
        score = metric.compute(predictions, references)
        assert score == 0.0
    
    def test_f1_empty_reference(self):
        """Test F1 score with empty reference."""
        metric = F1Score()
        predictions = ["hello world"]
        references = [""]
        
        score = metric.compute(predictions, references)
        assert score == 0.0
    
    def test_f1_both_empty(self):
        """Test F1 score with both prediction and reference empty."""
        metric = F1Score()
        predictions = [""]
        references = [""]
        
        score = metric.compute(predictions, references)
        assert score == 1.0
    
    def test_f1_multiple_predictions(self):
        """Test F1 score with multiple predictions."""
        metric = F1Score()
        predictions = ["hello world", "test case", "good example"]
        references = ["hello world", "test example", "good case"]
        
        score = metric.compute(predictions, references)
        
        # Should be average of individual F1 scores
        assert 0 < score < 1


class TestRougeL:
    """Test cases for RougeL metric."""
    
    def test_rouge_l_perfect_match(self):
        """Test ROUGE-L with perfect match."""
        metric = RougeL()
        predictions = ["hello world test"]
        references = ["hello world test"]
        
        score = metric.compute(predictions, references)
        assert score == 1.0
    
    def test_rouge_l_no_common_subsequence(self):
        """Test ROUGE-L with no common subsequence."""
        metric = RougeL()
        predictions = ["abc def"]
        references = ["xyz uvw"]
        
        score = metric.compute(predictions, references)
        assert score == 0.0
    
    def test_rouge_l_partial_subsequence(self):
        """Test ROUGE-L with partial common subsequence."""
        metric = RougeL()
        predictions = ["hello world test case"]
        references = ["hello test example"]
        
        score = metric.compute(predictions, references)
        
        # LCS should be "hello test" (length 2)
        # Precision = 2/4, Recall = 2/3
        expected_precision = 2/4
        expected_recall = 2/3
        expected_f1 = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)
        
        assert abs(score - expected_f1) < 1e-6
    
    def test_rouge_l_empty_inputs(self):
        """Test ROUGE-L with empty inputs."""
        metric = RougeL()
        
        # Both empty
        assert metric.compute([""], [""]) == 1.0
        
        # One empty
        assert metric.compute(["hello"], [""]) == 0.0
        assert metric.compute([""], ["hello"]) == 0.0


class TestBLEU:
    """Test cases for BLEU metric."""
    
    def test_bleu_perfect_match(self):
        """Test BLEU with perfect match."""
        metric = BLEU(n_gram=2)
        predictions = ["hello world test case"]
        references = ["hello world test case"]
        
        score = metric.compute(predictions, references)
        assert score == 1.0
    
    def test_bleu_no_match(self):
        """Test BLEU with no n-gram matches."""
        metric = BLEU(n_gram=2)
        predictions = ["abc def ghi"]
        references = ["xyz uvw rst"]
        
        score = metric.compute(predictions, references)
        assert score == 0.0
    
    def test_bleu_partial_match(self):
        """Test BLEU with partial n-gram matches."""
        metric = BLEU(n_gram=2)
        predictions = ["hello world test"]
        references = ["hello world example"]
        
        score = metric.compute(predictions, references)
        
        # Should have some overlap in n-grams
        assert 0 < score < 1
    
    def test_bleu_empty_prediction(self):
        """Test BLEU with empty prediction."""
        metric = BLEU()
        predictions = [""]
        references = ["hello world"]
        
        score = metric.compute(predictions, references)
        assert score == 0.0
    
    def test_bleu_brevity_penalty(self):
        """Test BLEU brevity penalty."""
        metric = BLEU(n_gram=1)
        
        # Short prediction should have lower score due to brevity penalty
        short_predictions = ["hello"]
        long_predictions = ["hello world"]
        references = ["hello world test"]
        
        short_score = metric.compute(short_predictions, references)
        long_score = metric.compute(long_predictions, references)
        
        assert short_score < long_score


class TestCodeExecutionMetric:
    """Test cases for CodeExecutionMetric."""
    
    def test_code_execution_success(self):
        """Test successful code execution."""
        metric = CodeExecutionMetric(timeout=2.0)
        predictions = ["print('hello world')"]
        references = [{"expected_output": "hello world"}]
        
        score = metric.compute(predictions, references)
        assert score == 1.0
    
    def test_code_execution_syntax_error(self):
        """Test code with syntax error."""
        metric = CodeExecutionMetric(timeout=2.0)
        predictions = ["print('hello world'"]  # Syntax error
        references = [{"expected_output": "hello world"}]
        
        score = metric.compute(predictions, references)
        assert score == 0.0
    
    def test_code_execution_runtime_error(self):
        """Test code with runtime error."""
        metric = CodeExecutionMetric(timeout=2.0)
        predictions = ["x = 1/0"]  # Runtime error
        references = [{}]
        
        score = metric.compute(predictions, references)
        assert score == 0.0
    
    def test_code_execution_no_expected_output(self):
        """Test code execution without expected output (just success/failure)."""
        metric = CodeExecutionMetric(timeout=2.0)
        predictions = ["x = 5\ny = x * 2"]
        references = [{}]
        
        score = metric.compute(predictions, references)
        assert score == 1.0
    
    def test_code_execution_with_code_blocks(self):
        """Test code extraction from markdown code blocks."""
        metric = CodeExecutionMetric(timeout=2.0)
        predictions = ["```python\nprint('test')\n```"]
        references = [{}]
        
        score = metric.compute(predictions, references)
        assert score == 1.0
    
    def test_code_execution_timeout(self):
        """Test code execution timeout."""
        metric = CodeExecutionMetric(timeout=0.1)
        predictions = ["import time\ntime.sleep(1)"]  # Should timeout
        references = [{}]
        
        score = metric.compute(predictions, references)
        assert score == 0.0


@pytest.mark.unit
class TestMetricBatchCompute:
    """Test batch computation for all metrics."""
    
    def test_exact_match_batch(self):
        """Test ExactMatch batch_compute method."""
        metric = ExactMatch()
        predictions = ["hello", "world", "test"]
        references = ["hello", "world", "different"]
        
        batch_scores = metric.batch_compute(predictions, references)
        
        assert len(batch_scores) == 3
        assert batch_scores[0] == 1.0  # "hello" == "hello"
        assert batch_scores[1] == 1.0  # "world" == "world"
        assert batch_scores[2] == 0.0  # "test" != "different"
    
    def test_f1_score_batch(self):
        """Test F1Score batch_compute method."""
        metric = F1Score()
        predictions = ["hello world", "test case", "example text"]
        references = ["hello world", "test different", "example text"]
        
        batch_scores = metric.batch_compute(predictions, references)
        
        assert len(batch_scores) == 3
        assert batch_scores[0] == 1.0  # Perfect match
        assert 0 < batch_scores[1] < 1.0  # Partial match
        assert batch_scores[2] == 1.0  # Perfect match