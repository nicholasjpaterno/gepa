#!/usr/bin/env python3
"""
Custom Metrics Example

This example demonstrates how to create and use custom evaluation metrics with GEPA.
We'll build a domain-specific email classification system with custom business metrics.

Run with: python examples/custom_metrics.py
"""

import asyncio
import os
import re
from typing import List, Dict, Any, Union
from abc import ABC, abstractmethod

from gepa import GEPAOptimizer, GEPAConfig
from gepa.core.system import CompoundAISystem, LanguageModule, SequentialFlow, IOSchema
from gepa.evaluation.base import Metric, SimpleEvaluator
from gepa.evaluation.metrics import ExactMatch, F1Score
from gepa.inference.factory import InferenceFactory


# Custom metrics for email classification
class PriorityAccuracy(Metric):
    """Custom metric that heavily weights correct high-priority classifications."""
    
    def __init__(self, name: str = "priority_accuracy"):
        super().__init__(name)
        self.priority_weights = {
            "urgent": 3.0,    # High weight for urgent emails
            "high": 2.0,      # Medium weight for high priority
            "normal": 1.0,    # Standard weight for normal
            "low": 0.5        # Lower weight for low priority
        }
    
    def compute(self, predictions: List[str], references: List[str]) -> float:
        """Compute weighted accuracy based on priority levels."""
        if not predictions or not references or len(predictions) != len(references):
            return 0.0
        
        total_weight = 0.0
        correct_weight = 0.0
        
        for pred, ref in zip(predictions, references):
            # Extract priority level from reference
            ref_priority = ref.lower().strip()
            pred_priority = pred.lower().strip()
            
            # Get weight for this priority level
            weight = self.priority_weights.get(ref_priority, 1.0)
            total_weight += weight
            
            # Add weight if prediction is correct
            if pred_priority == ref_priority:
                correct_weight += weight
        
        return correct_weight / total_weight if total_weight > 0 else 0.0


class BusinessImpactScore(Metric):
    """Custom metric measuring business impact of misclassifications."""
    
    def __init__(self, name: str = "business_impact"):
        super().__init__(name)
        # Cost matrix: [actual][predicted] = cost of misclassification
        self.cost_matrix = {
            "urgent": {"urgent": 0, "high": 1, "normal": 5, "low": 10},
            "high": {"urgent": 0, "high": 0, "normal": 2, "low": 5},
            "normal": {"urgent": 0, "high": 0, "normal": 0, "low": 1},
            "low": {"urgent": 0, "high": 0, "normal": 0, "low": 0}
        }
    
    def compute(self, predictions: List[str], references: List[str]) -> float:
        """Compute business impact score (higher is better)."""
        if not predictions or not references or len(predictions) != len(references):
            return 0.0
        
        total_cost = 0.0
        max_possible_cost = 0.0
        
        for pred, ref in zip(predictions, references):
            ref_clean = ref.lower().strip()
            pred_clean = pred.lower().strip()
            
            # Get actual cost
            if ref_clean in self.cost_matrix and pred_clean in self.cost_matrix[ref_clean]:
                cost = self.cost_matrix[ref_clean][pred_clean]
            else:
                cost = 5  # Default high cost for unknown classifications
            
            total_cost += cost
            
            # Maximum cost would be misclassifying as lowest priority
            max_cost = max(self.cost_matrix[ref_clean].values()) if ref_clean in self.cost_matrix else 5
            max_possible_cost += max_cost
        
        # Return normalized score (1.0 - cost_ratio)
        if max_possible_cost == 0:
            return 1.0
        
        cost_ratio = total_cost / max_possible_cost
        return max(0.0, 1.0 - cost_ratio)


class ResponseTimeMetric(Metric):
    """Custom metric for evaluating response time predictions."""
    
    def __init__(self, name: str = "response_time_accuracy"):
        super().__init__(name)
    
    def compute(self, predictions: List[str], references: List[str]) -> float:
        """Compute accuracy of response time predictions."""
        if not predictions or not references or len(predictions) != len(references):
            return 0.0
        
        correct = 0
        total = 0
        
        for pred, ref in zip(predictions, references):
            # Extract time values (e.g., "4 hours", "1 day", "immediate")
            pred_time = self._parse_time(pred)
            ref_time = self._parse_time(ref)
            
            if pred_time is not None and ref_time is not None:
                # Allow 25% tolerance in time predictions
                tolerance = ref_time * 0.25
                if abs(pred_time - ref_time) <= tolerance:
                    correct += 1
            elif pred == ref:  # Exact string match for special cases like "immediate"
                correct += 1
            
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _parse_time(self, time_str: str) -> Union[float, None]:
        """Parse time string to hours."""
        time_str = time_str.lower().strip()
        
        if "immediate" in time_str:
            return 0.0
        
        # Extract numbers and units
        hour_match = re.search(r'(\d+)\s*hour', time_str)
        day_match = re.search(r'(\d+)\s*day', time_str)
        minute_match = re.search(r'(\d+)\s*minute', time_str)
        
        if day_match:
            return float(day_match.group(1)) * 24
        elif hour_match:
            return float(hour_match.group(1))
        elif minute_match:
            return float(minute_match.group(1)) / 60
        
        return None


# Sample email dataset with business context
EMAIL_DATASET = [
    {
        "email": "URGENT: Server down! All systems offline. Need immediate attention!",
        "sender": "admin@company.com",
        "subject": "CRITICAL: System Outage",
        "expected_priority": "urgent",
        "expected_response_time": "immediate"
    },
    {
        "email": "Hi team, could you please review the quarterly report when you have time? Thanks!",
        "sender": "manager@company.com", 
        "subject": "Quarterly Report Review",
        "expected_priority": "normal",
        "expected_response_time": "2 days"
    },
    {
        "email": "The printer in room 205 is out of toner. Can someone replace it?",
        "sender": "office@company.com",
        "subject": "Printer Maintenance",
        "expected_priority": "low",
        "expected_response_time": "1 day"
    },
    {
        "email": "Client meeting scheduled for tomorrow 9 AM. Need presentation slides ready.",
        "sender": "sales@company.com",
        "subject": "Client Meeting Tomorrow",
        "expected_priority": "high", 
        "expected_response_time": "4 hours"
    },
    {
        "email": "Security breach detected! Unauthorized access to customer data. Act now!",
        "sender": "security@company.com",
        "subject": "SECURITY ALERT",
        "expected_priority": "urgent",
        "expected_response_time": "immediate"
    },
    {
        "email": "Weekly team lunch is moved to Friday. Please update your calendars.",
        "sender": "hr@company.com",
        "subject": "Team Lunch Update",
        "expected_priority": "low",
        "expected_response_time": "1 day"
    },
    {
        "email": "Budget approval needed for Q4 marketing campaign. Deadline is next week.",
        "sender": "marketing@company.com",
        "subject": "Budget Approval Required",
        "expected_priority": "high",
        "expected_response_time": "1 day"
    }
]


def create_email_classification_system() -> CompoundAISystem:
    """Create an email classification and response time prediction system."""
    
    return CompoundAISystem(
        modules={
            "analyzer": LanguageModule(
                id="analyzer",
                prompt="""Analyze this email and classify its priority level and required response time.

Email: {email}
From: {sender}
Subject: {subject}

Classify the priority as: urgent, high, normal, or low
Estimate response time as: immediate, 2 hours, 4 hours, 1 day, 2 days, or 1 week

Priority: 
Response Time:""",
                model_weights="gpt-3.5-turbo"
            )
        },
        control_flow=SequentialFlow(["analyzer"]),
        input_schema=IOSchema(
            fields={"email": str, "sender": str, "subject": str},
            required=["email", "sender", "subject"]
        ),
        output_schema=IOSchema(
            fields={"priority": str, "response_time": str},
            required=["priority", "response_time"]
        ),
        system_id="email_classifier"
    )


def create_dataset_for_gepa(email_data: List[Dict]) -> List[Dict[str, Any]]:
    """Convert email data to GEPA format."""
    dataset = []
    
    for item in email_data:
        dataset.append({
            "email": item["email"],
            "sender": item["sender"],
            "subject": item["subject"],
            "expected_priority": item["expected_priority"],
            "expected_response_time": item["expected_response_time"]
        })
    
    return dataset


class MultiOutputEvaluator(SimpleEvaluator):
    """Custom evaluator that handles multiple outputs (priority and response time)."""
    
    def __init__(self, priority_metrics: List[Metric], response_time_metrics: List[Metric]):
        # Combine all metrics
        all_metrics = priority_metrics + response_time_metrics
        super().__init__(all_metrics)
        self.priority_metrics = priority_metrics
        self.response_time_metrics = response_time_metrics
    
    async def evaluate_system(
        self, 
        system: CompoundAISystem, 
        dataset: List[Dict[str, Any]], 
        inference_client
    ) -> Dict[str, float]:
        """Evaluate system with separate handling for priority and response time."""
        
        # Generate predictions (this would normally call the system)
        priority_predictions = []
        response_time_predictions = []
        priority_references = []
        response_time_references = []
        
        for item in dataset:
            # In practice, you'd call: result = await system.execute(item, inference_client)
            # For demo, we'll simulate predictions
            priority_predictions.append("normal")  # Placeholder
            response_time_predictions.append("1 day")  # Placeholder
            
            priority_references.append(item["expected_priority"])
            response_time_references.append(item["expected_response_time"])
        
        results = {}
        
        # Evaluate priority metrics
        for metric in self.priority_metrics:
            score = metric.compute(priority_predictions, priority_references)
            results[metric.name] = score
        
        # Evaluate response time metrics
        for metric in self.response_time_metrics:
            score = metric.compute(response_time_predictions, response_time_references)
            results[metric.name] = score
        
        return results


async def main():
    """Run the custom metrics example."""
    
    print("📧 GEPA Custom Metrics for Email Classification")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    # 1. Create the email classification system
    print("🏗️  Creating email classification system...")
    system = create_email_classification_system()
    
    print(f"   System: {system.system_id}")
    print("   Outputs: Priority level + Response time estimate")
    print()
    
    # 2. Prepare dataset
    print("📊 Preparing email dataset...")
    dataset = create_dataset_for_gepa(EMAIL_DATASET)
    
    print(f"   Emails: {len(dataset)}")
    print("   Sample priorities:")
    for priority in ["urgent", "high", "normal", "low"]:
        count = sum(1 for item in dataset if item["expected_priority"] == priority)
        print(f"     {priority}: {count} emails")
    print()
    
    # 3. Create custom metrics
    print("🎯 Setting up custom business metrics...")
    
    # Priority-focused metrics
    priority_metrics = [
        ExactMatch(name="priority_exact"),
        PriorityAccuracy(name="priority_weighted"),
        BusinessImpactScore(name="business_impact")
    ]
    
    # Response time metrics
    response_time_metrics = [
        ExactMatch(name="response_exact"),
        ResponseTimeMetric(name="response_accuracy")
    ]
    
    print("   Priority Metrics:")
    print("     - Exact Match: Standard accuracy")
    print("     - Priority Weighted: Higher weight for urgent emails")
    print("     - Business Impact: Cost-based scoring")
    print()
    print("   Response Time Metrics:")
    print("     - Exact Match: Exact time match")
    print("     - Response Accuracy: 25% tolerance for time estimates")
    print()
    
    # 4. Create custom evaluator
    evaluator = MultiOutputEvaluator(priority_metrics, response_time_metrics)
    
    # 5. Configure GEPA
    print("⚙️  Configuring GEPA...")
    
    config = GEPAConfig(
        inference={
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "api_key": api_key,
            "max_tokens": 100,
            "temperature": 0.1
        },
        optimization={
            "budget": 20,
            "pareto_set_size": 6,
            "minibatch_size": 3,
            "enable_crossover": True,
            "mutation_types": ["rewrite", "insert"]
        },
        database={
            "url": "sqlite:///gepa_custom_metrics.db"
        }
    )
    
    # 6. Create inference client
    inference_client = InferenceFactory.create_client(config.inference)
    
    # 7. Demonstrate custom metrics
    print("🧪 Testing custom metrics with sample data...")
    
    # Sample predictions for demonstration
    sample_priority_preds = ["urgent", "normal", "low", "high", "urgent", "normal", "high"]
    sample_priority_refs = ["urgent", "normal", "low", "high", "urgent", "low", "high"]
    
    sample_time_preds = ["immediate", "2 days", "1 day", "3 hours", "immediate", "1 day", "1 day"]
    sample_time_refs = ["immediate", "2 days", "1 day", "4 hours", "immediate", "1 day", "1 day"]
    
    print("   Priority Classification Results:")
    for metric in priority_metrics:
        score = metric.compute(sample_priority_preds, sample_priority_refs)
        print(f"     {metric.name}: {score:.3f}")
    
    print()
    print("   Response Time Prediction Results:")
    for metric in response_time_metrics:
        score = metric.compute(sample_time_preds, sample_time_refs)
        print(f"     {metric.name}: {score:.3f}")
    
    print()
    
    # 8. Explain metric differences
    print("💡 Custom Metrics Insights:")
    print("-" * 30)
    
    # Show business impact calculation
    business_metric = BusinessImpactScore()
    business_score = business_metric.compute(sample_priority_preds, sample_priority_refs)
    
    print(f"Business Impact Score: {business_score:.3f}")
    print("   - Penalizes misclassifying urgent emails as low priority")
    print("   - Lower cost for conservative over-classification")
    print("   - Reflects real business consequences")
    print()
    
    # Show priority weighting
    priority_metric = PriorityAccuracy()
    priority_score = priority_metric.compute(sample_priority_preds, sample_priority_refs)
    
    print(f"Priority Weighted Score: {priority_score:.3f}")
    print("   - 3x weight for urgent email accuracy")
    print("   - 2x weight for high priority emails")
    print("   - Standard weight for normal priority")
    print("   - 0.5x weight for low priority emails")
    print()
    
    # 9. Run optimization (simplified for demo)
    print("🔄 Running optimization with custom metrics...")
    print("   (In practice, this would optimize the email classification system)")
    
    # Create optimizer
    optimizer = GEPAOptimizer(
        config=config,
        evaluator=evaluator,
        inference_client=inference_client
    )
    
    try:
        # For demo purposes, we'll simulate optimization results
        print("   Optimization would proceed with custom metrics...")
        print("   Business impact metric guides toward safer classifications")
        print("   Priority weighting focuses on critical email accuracy")
        print("   Response time metric balances speed with accuracy")
        
        # Show how custom metrics affect optimization
        print()
        print("📈 Custom Metrics Impact on Optimization:")
        print("   - Business Impact: Favors systems that avoid costly mistakes")
        print("   - Priority Weighting: Improves urgent/high priority accuracy")
        print("   - Response Time: Balances realistic vs. optimal estimates")
        print("   - Multi-objective: Pareto frontier includes all trade-offs")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Clean up
        if hasattr(inference_client, 'close'):
            await inference_client.close()
    
    # 10. Guidelines for custom metrics
    print("\n🎯 Custom Metrics Best Practices:")
    print("-" * 35)
    print("1. Domain Relevance: Reflect real business/user needs")
    print("2. Interpretability: Clear meaning for stakeholders")
    print("3. Robustness: Handle edge cases and invalid inputs")
    print("4. Normalization: Return scores in consistent ranges (0-1)")
    print("5. Efficiency: Fast computation for large datasets")
    print("6. Composability: Work well with other metrics in Pareto optimization")
    
    print("\n🔧 Implementation Tips:")
    print("- Inherit from gepa.evaluation.base.Metric")
    print("- Implement compute() and optionally batch_compute()")
    print("- Add comprehensive error handling")
    print("- Include docstrings explaining the metric's purpose")
    print("- Test with edge cases (empty lists, mismatched lengths)")
    
    print("\n🎉 Custom metrics example completed!")
    
    print("\nKey learnings:")
    print("- Custom metrics enable domain-specific optimization")
    print("- Business constraints can be encoded as metrics")
    print("- Multi-output systems need specialized evaluators")
    print("- GEPA's Pareto optimization handles trade-offs naturally")
    
    print("\nNext examples:")
    print("- examples/advanced_system.py - Complex multi-agent workflows")
    print("- examples/production_deployment.py - Full production setup")


if __name__ == "__main__":
    asyncio.run(main())