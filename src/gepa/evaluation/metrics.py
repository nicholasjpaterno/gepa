"""Common evaluation metrics for GEPA."""

import re
import string
from typing import Any, List, Set
from collections import Counter

from .base import Metric


class ExactMatch(Metric):
    """Exact match metric."""
    
    def __init__(self, name: str = "exact_match", case_sensitive: bool = False):
        super().__init__(name)
        self.case_sensitive = case_sensitive
    
    def compute(self, predictions: List[Any], references: List[Any]) -> float:
        """Compute exact match score."""
        if len(predictions) != len(references):
            return 0.0
        
        matches = 0
        for pred, ref in zip(predictions, references):
            pred_str = str(pred).strip()
            ref_str = str(ref).strip()
            
            if not self.case_sensitive:
                pred_str = pred_str.lower()
                ref_str = ref_str.lower()
            
            if pred_str == ref_str:
                matches += 1
        
        return matches / len(predictions)


class F1Score(Metric):
    """F1 score based on token overlap."""
    
    def __init__(self, name: str = "f1_score"):
        super().__init__(name)
    
    def compute(self, predictions: List[Any], references: List[Any]) -> float:
        """Compute F1 score."""
        if len(predictions) != len(references):
            return 0.0
        
        f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = self._tokenize(str(pred))
            ref_tokens = self._tokenize(str(ref))
            
            f1 = self._compute_f1(pred_tokens, ref_tokens)
            f1_scores.append(f1)
        
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Remove punctuation and convert to lowercase
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()
    
    def _compute_f1(self, pred_tokens: List[str], ref_tokens: List[str]) -> float:
        """Compute F1 score between two token lists."""
        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)
        
        # Calculate overlap
        overlap = 0
        for token in pred_counter:
            overlap += min(pred_counter[token], ref_counter.get(token, 0))
        
        if overlap == 0:
            return 0.0
        
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1


class RougeL(Metric):
    """ROUGE-L metric based on longest common subsequence."""
    
    def __init__(self, name: str = "rouge_l"):
        super().__init__(name)
    
    def compute(self, predictions: List[Any], references: List[Any]) -> float:
        """Compute ROUGE-L score."""
        if len(predictions) != len(references):
            return 0.0
        
        rouge_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = str(pred).split()
            ref_tokens = str(ref).split()
            
            rouge = self._compute_rouge_l(pred_tokens, ref_tokens)
            rouge_scores.append(rouge)
        
        return sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    
    def _compute_rouge_l(self, pred_tokens: List[str], ref_tokens: List[str]) -> float:
        """Compute ROUGE-L score using LCS."""
        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        lcs_length = self._lcs_length(pred_tokens, ref_tokens)
        
        if lcs_length == 0:
            return 0.0
        
        precision = lcs_length / len(pred_tokens)
        recall = lcs_length / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]


class BLEU(Metric):
    """BLEU score metric."""
    
    def __init__(self, name: str = "bleu", n_gram: int = 4):
        super().__init__(name)
        self.n_gram = n_gram
    
    def compute(self, predictions: List[Any], references: List[Any]) -> float:
        """Compute BLEU score."""
        if len(predictions) != len(references):
            return 0.0
        
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = str(pred).split()
            ref_tokens = str(ref).split()
            
            bleu = self._compute_bleu(pred_tokens, ref_tokens)
            bleu_scores.append(bleu)
        
        return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    
    def _compute_bleu(self, pred_tokens: List[str], ref_tokens: List[str]) -> float:
        """Compute BLEU score for single prediction-reference pair."""
        if not pred_tokens:
            return 0.0
        
        # Compute n-gram precisions
        precisions = []
        for n in range(1, min(self.n_gram + 1, len(pred_tokens) + 1)):
            pred_ngrams = self._get_ngrams(pred_tokens, n)
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            
            if not pred_ngrams:
                continue
            
            # Count matches
            matches = 0
            for ngram in pred_ngrams:
                if ngram in ref_ngrams:
                    matches += min(pred_ngrams[ngram], ref_ngrams[ngram])
            
            precision = matches / sum(pred_ngrams.values())
            precisions.append(precision)
        
        if not precisions:
            return 0.0
        
        # Geometric mean of precisions
        import math
        geo_mean = math.exp(sum(math.log(p) for p in precisions if p > 0) / len(precisions))
        
        # Brevity penalty
        pred_len = len(pred_tokens)
        ref_len = len(ref_tokens)
        
        if pred_len >= ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - ref_len / pred_len)
        
        return bp * geo_mean
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-grams from token list."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        return Counter(ngrams)


class CodeExecutionMetric(Metric):
    """Metric for evaluating code execution success."""
    
    def __init__(self, name: str = "code_execution", timeout: float = 5.0):
        super().__init__(name)
        self.timeout = timeout
    
    def compute(self, predictions: List[Any], references: List[Any]) -> float:
        """Compute code execution success rate."""
        if len(predictions) != len(references):
            return 0.0
        
        successes = 0
        for pred, ref in zip(predictions, references):
            try:
                # Extract code from prediction
                code = self._extract_code(str(pred))
                
                # Get expected output from reference
                expected = ref.get("expected_output") if isinstance(ref, dict) else None
                
                # Execute code
                success = self._execute_code(code, expected)
                if success:
                    successes += 1
                    
            except Exception:
                # Execution failed
                pass
        
        return successes / len(predictions)
    
    def _extract_code(self, text: str) -> str:
        """Extract code from text (assuming it's in code blocks)."""
        # Look for code blocks
        code_pattern = r'```(?:python)?\s*(.*?)```'
        matches = re.findall(code_pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks, return the whole text
        return text.strip()
    
    def _execute_code(self, code: str, expected_output: Any = None) -> bool:
        """Execute code and check if it runs successfully."""
        import subprocess
        import tempfile
        import os
        
        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute with timeout
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # Clean up
            os.unlink(temp_file)
            
            # Check if execution was successful
            if result.returncode != 0:
                return False
            
            # If expected output is provided, check it matches
            if expected_output is not None:
                actual_output = result.stdout.strip()
                return actual_output == str(expected_output).strip()
            
            return True
            
        except Exception:
            return False


class SemanticSimilarity(Metric):
    """Semantic similarity metric using embeddings."""
    
    def __init__(self, name: str = "semantic_similarity", model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(name)
        self.model_name = model_name
        self._model = None
    
    def _get_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("sentence-transformers package required for SemanticSimilarity metric")
        return self._model
    
    def compute(self, predictions: List[Any], references: List[Any]) -> float:
        """Compute semantic similarity score."""
        if len(predictions) != len(references):
            return 0.0
        
        model = self._get_model()
        
        pred_texts = [str(pred) for pred in predictions]
        ref_texts = [str(ref) for ref in references]
        
        # Get embeddings
        pred_embeddings = model.encode(pred_texts)
        ref_embeddings = model.encode(ref_texts)
        
        # Compute cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = []
        
        for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
            sim = cosine_similarity([pred_emb], [ref_emb])[0][0]
            similarities.append(max(0.0, sim))  # Ensure non-negative
        
        return sum(similarities) / len(similarities)