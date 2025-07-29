"""Command-line interface for GEPA."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from .config import GEPAConfig
from .core.optimizer import GEPAOptimizer
from .core.system import CompoundAISystem, LanguageModule, SequentialFlow, IOSchema
from .evaluation.base import SimpleEvaluator
from .evaluation.metrics import ExactMatch, F1Score

app = typer.Typer(help="GEPA: Reflective Prompt Evolution")
console = Console()


@app.command()
def optimize(
    config_path: Path = typer.Argument(..., help="Path to GEPA configuration file"),
    system_path: Path = typer.Argument(..., help="Path to system definition file"),
    dataset_path: Path = typer.Argument(..., help="Path to training dataset"),
    output_path: Optional[Path] = typer.Option(None, help="Path to save results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Optimize a compound AI system using GEPA."""
    asyncio.run(_optimize_async(config_path, system_path, dataset_path, output_path, verbose))


async def _optimize_async(
    config_path: Path,
    system_path: Path, 
    dataset_path: Path,
    output_path: Optional[Path],
    verbose: bool
):
    """Async optimization function."""
    try:
        # Load configuration
        console.print(f"Loading configuration from {config_path}")
        config = GEPAConfig.from_file(config_path)
        
        # Load system definition
        console.print(f"Loading system definition from {system_path}")
        system = _load_system(system_path)
        
        # Load dataset
        console.print(f"Loading dataset from {dataset_path}")
        dataset = _load_dataset(dataset_path)
        
        # Create evaluator
        evaluator = SimpleEvaluator([
            ExactMatch(),
            F1Score()
        ])
        
        # Create optimizer
        optimizer = GEPAOptimizer(config, evaluator)
        
        # Run optimization
        console.print("\n[bold green]Starting GEPA optimization...[/bold green]")
        
        with Progress() as progress:
            task = progress.add_task(
                f"Optimizing (budget: {config.optimization.budget})", 
                total=config.optimization.budget
            )
            
            # This is a simplified progress tracking - in reality you'd want
            # to update this during optimization
            result = await optimizer.optimize(system, dataset)
            progress.update(task, completed=config.optimization.budget)
        
        # Display results
        _display_results(result, verbose)
        
        # Save results if requested
        if output_path:
            _save_results(result, output_path)
            console.print(f"Results saved to {output_path}")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def _load_system(system_path: Path) -> CompoundAISystem:
    """Load system definition from file."""
    with open(system_path) as f:
        system_config = yaml.safe_load(f)
    
    # Create modules
    modules = {}
    for module_config in system_config["modules"]:
        module = LanguageModule(
            id=module_config["id"],
            prompt=module_config["prompt"],
            model_weights=module_config.get("model")
        )
        modules[module.id] = module
    
    # Create control flow
    flow_config = system_config.get("control_flow", {})
    if flow_config.get("type") == "sequential":
        control_flow = SequentialFlow(flow_config["module_order"])
    else:
        # Default to sequential with all modules
        control_flow = SequentialFlow(list(modules.keys()))
    
    # Create schemas
    input_schema = IOSchema(
        fields=system_config.get("input_schema", {}).get("fields", {}),
        required=system_config.get("input_schema", {}).get("required", [])
    )
    
    output_schema = IOSchema(
        fields=system_config.get("output_schema", {}).get("fields", {}),
        required=system_config.get("output_schema", {}).get("required", [])
    )
    
    return CompoundAISystem(
        modules=modules,
        control_flow=control_flow,
        input_schema=input_schema,
        output_schema=output_schema,
        system_id=system_config.get("id", "default")
    )


def _load_dataset(dataset_path: Path) -> list:
    """Load dataset from file."""
    with open(dataset_path) as f:
        if dataset_path.suffix == ".json":
            import json
            return json.load(f)
        else:
            return yaml.safe_load(f)


def _display_results(result, verbose: bool):
    """Display optimization results."""
    console.print("\n[bold green]Optimization Results[/bold green]")
    
    # Summary table
    table = Table(title="Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Best Score", f"{result.best_score:.4f}")
    table.add_row("Total Rollouts", str(result.total_rollouts))
    table.add_row("Total Cost", f"${result.total_cost:.4f}")
    table.add_row("Pareto Frontier Size", str(result.pareto_frontier.size()))
    
    console.print(table)
    
    if verbose:
        # Pareto frontier details
        console.print("\n[bold cyan]Pareto Frontier[/bold cyan]")
        frontier_table = Table()
        frontier_table.add_column("Candidate ID", style="cyan")
        frontier_table.add_column("Scores", style="green")
        frontier_table.add_column("Cost", style="yellow")
        
        for candidate in result.pareto_frontier.candidates:
            scores_str = ", ".join(f"{k}: {v:.3f}" for k, v in candidate.scores.items())
            frontier_table.add_row(
                candidate.id[:8] + "...",
                scores_str,
                f"${candidate.cost:.4f}"
            )
        
        console.print(frontier_table)


def _save_results(result, output_path: Path):
    """Save results to file."""
    # Convert result to serializable format
    result_data = {
        "best_score": result.best_score,
        "total_rollouts": result.total_rollouts,
        "total_cost": result.total_cost,
        "pareto_frontier": {
            "size": result.pareto_frontier.size(),
            "candidates": [
                {
                    "id": candidate.id,
                    "scores": candidate.scores,
                    "cost": candidate.cost,
                    "tokens_used": candidate.tokens_used,
                    "generation": candidate.generation
                }
                for candidate in result.pareto_frontier.candidates
            ]
        },
        "optimization_history": result.optimization_history
    }
    
    with open(output_path, 'w') as f:
        if output_path.suffix == ".json":
            import json
            json.dump(result_data, f, indent=2)
        else:
            yaml.dump(result_data, f, default_flow_style=False)


@app.command()
def validate_config(
    config_path: Path = typer.Argument(..., help="Path to configuration file")
):
    """Validate a GEPA configuration file."""
    try:
        config = GEPAConfig.from_file(config_path)
        console.print("[green]Configuration is valid![/green]")
        
        # Display config summary
        table = Table(title="Configuration Summary")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Provider", config.inference.provider)
        table.add_row("Model", config.inference.model)
        table.add_row("Budget", str(config.optimization.budget))
        table.add_row("Pareto Set Size", str(config.optimization.pareto_set_size))
        table.add_row("Enable Crossover", str(config.optimization.enable_crossover))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Configuration is invalid: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def create_example(
    output_dir: Path = typer.Option("./gepa_example", help="Output directory for examples")
):
    """Create example configuration and system files."""
    output_dir.mkdir(exist_ok=True)
    
    # Create example config
    example_config = {
        "inference": {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "api_key": "${OPENAI_API_KEY}",
            "max_tokens": 4096,
            "temperature": 0.7,
            "timeout": 30.0,
            "retry_attempts": 3
        },
        "optimization": {
            "budget": 100,
            "minibatch_size": 5,
            "pareto_set_size": 10,
            "enable_crossover": True,
            "crossover_probability": 0.3,
            "mutation_types": ["rewrite", "insert", "delete", "compress"]
        }
    }
    
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False)
    
    # Create example system
    example_system = {
        "id": "qa_system",
        "modules": [
            {
                "id": "answerer",
                "prompt": "Answer the following question based on the given context.\n\nContext: {context}\nQuestion: {question}\n\nAnswer:",
                "model": "gpt-3.5-turbo"
            }
        ],
        "control_flow": {
            "type": "sequential",
            "module_order": ["answerer"]
        },
        "input_schema": {
            "fields": {
                "context": "str",
                "question": "str"
            },
            "required": ["question"]
        },
        "output_schema": {
            "fields": {
                "answer": "str"
            },
            "required": ["answer"]
        }
    }
    
    system_path = output_dir / "system.yaml"
    with open(system_path, 'w') as f:
        yaml.dump(example_system, f, default_flow_style=False)
    
    # Create example dataset
    example_dataset = [
        {
            "context": "The capital of France is Paris.",
            "question": "What is the capital of France?",
            "expected": "Paris"
        },
        {
            "context": "Python is a programming language.",
            "question": "What is Python?",
            "expected": "A programming language"
        }
    ]
    
    dataset_path = output_dir / "dataset.yaml"
    with open(dataset_path, 'w') as f:
        yaml.dump(example_dataset, f, default_flow_style=False)
    
    console.print(f"[green]Example files created in {output_dir}[/green]")
    console.print("\nFiles created:")
    console.print(f"- {config_path} (configuration)")
    console.print(f"- {system_path} (system definition)")
    console.print(f"- {dataset_path} (example dataset)")
    console.print("\nTo run optimization:")
    console.print(f"gepa optimize {config_path} {system_path} {dataset_path}")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()