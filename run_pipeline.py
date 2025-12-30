#!/usr/bin/env python3
"""CLI entry point for the MetaGlyph pipeline.

Usage:
    # Run complete pipeline
    python run_pipeline.py

    # Run specific stages
    python run_pipeline.py --stage 1      # Just generate datasets
    python run_pipeline.py --stage 1-3    # Stages 1 through 3
    python run_pipeline.py --stage 4,5    # Stages 4 and 5

    # Configure models
    python run_pipeline.py --models llama3.2,mistral

    # Set instance count
    python run_pipeline.py --instances 100

    # Use config file
    python run_pipeline.py --config custom_config.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Load .env file if present
def load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import Pipeline, PipelineConfig


def parse_stages(stage_arg: str) -> list[int]:
    """Parse stage argument into list of stage numbers."""
    if not stage_arg:
        return list(range(1, 7))

    stages = []
    for part in stage_arg.split(','):
        if '-' in part:
            start, end = part.split('-')
            stages.extend(range(int(start), int(end) + 1))
        else:
            stages.append(int(part))

    return sorted(set(stages))


def main():
    parser = argparse.ArgumentParser(
        description="MetaGlyph Pipeline - Symbolic Metalanguages for LLM Prompting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration JSON file",
    )
    parser.add_argument(
        "--stage",
        type=str,
        help="Stage(s) to run (e.g., '1', '1-3', '4,5'). Default: all",
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models to use",
    )
    parser.add_argument(
        "--instances",
        type=int,
        help="Number of instances per task family",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["ollama", "openai", "anthropic", "openrouter"],
        help="Model backend to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Base output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
    else:
        config_path = Path(__file__).parent / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            config_dict = {}

    # Override with command line arguments
    if args.models:
        config_dict["models"] = args.models.split(',')
    if args.instances:
        config_dict["instances_per_family"] = args.instances
    if args.backend:
        config_dict["backend"] = args.backend
    if args.output_dir:
        config_dict["base_dir"] = args.output_dir
    if args.seed:
        config_dict["seed"] = args.seed

    # Create config
    config = PipelineConfig(**config_dict)

    # Create pipeline
    pipeline = Pipeline(config)

    # Determine stages to run
    stages = parse_stages(args.stage)

    print(f"MetaGlyph Pipeline")
    print(f"=" * 50)
    print(f"Stages to run: {stages}")
    print(f"Models: {config.models}")
    print(f"Instances per family: {config.instances_per_family}")
    print(f"Backend: {config.backend}")
    print(f"=" * 50)
    print()

    # Run stages
    results = {}
    for stage in stages:
        result = pipeline.run_stage(stage)
        results[f"stage_{stage}"] = result

        if not result.get("success", False):
            print(f"Stage {stage} failed: {result.get('error', 'Unknown error')}")
            if stage < max(stages):
                response = input("Continue with remaining stages? [y/N] ")
                if response.lower() != 'y':
                    break

    print()
    print(f"Pipeline completed")
    print(f"=" * 50)

    # Summary
    for stage in stages:
        key = f"stage_{stage}"
        if key in results:
            status = "OK" if results[key].get("success") else "FAILED"
            elapsed = results[key].get("elapsed_seconds", 0)
            print(f"Stage {stage}: {status} ({elapsed:.1f}s)")

    return 0 if all(r.get("success", False) for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
