"""MetaGlyph Pipeline Orchestrator.

This module provides the main entry point for running the complete
6-stage experimental pipeline.

Pipeline stages:
1. Dataset & task specification
2. Prompt construction
3. Token accounting & matching
4. Model execution
5. Automatic evaluation
6. Aggregation & reporting

Usage:
    from src.pipeline import Pipeline

    pipeline = Pipeline(config)
    pipeline.run_all()

Or run individual stages:
    pipeline.run_stage(1)  # Just generate datasets
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import logging
import time

from .stages import (
    DatasetGenerator,
    PromptConstructor,
    TokenMatcher,
    ModelExecutor,
    Evaluator,
    Aggregator,
)
from .stages.stage4_execution import ExecutionConfig
from .utils.io_utils import save_json, load_json, ensure_dir


@dataclass
class PipelineConfig:
    """Configuration for the entire pipeline."""
    # Directories
    base_dir: Path = field(default_factory=lambda: Path("."))
    tasks_dir: Optional[Path] = None
    prompts_dir: Optional[Path] = None
    tokens_dir: Optional[Path] = None
    outputs_dir: Optional[Path] = None
    runs_dir: Optional[Path] = None
    results_dir: Optional[Path] = None
    summary_dir: Optional[Path] = None

    # Dataset generation
    instances_per_family: int = 50
    seed: int = 42

    # Token matching
    token_tolerance: int = 1
    tokenizer_model: str = "simple"

    # Model execution
    models: list[str] = field(default_factory=lambda: ["llama3.2"])
    backend: str = "ollama"
    temperature: float = 0.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 781
    max_context_tokens: int = 8192
    request_delay: float = 0.5  # Seconds between requests (rate limit protection)
    skip_existing: bool = True

    def __post_init__(self):
        """Set default directories based on base_dir."""
        self.base_dir = Path(self.base_dir)

        if self.tasks_dir is None:
            self.tasks_dir = self.base_dir / "tasks"
        if self.prompts_dir is None:
            self.prompts_dir = self.base_dir / "prompts"
        if self.tokens_dir is None:
            self.tokens_dir = self.base_dir / "tokens"
        if self.outputs_dir is None:
            self.outputs_dir = self.base_dir / "outputs"
        if self.runs_dir is None:
            self.runs_dir = self.base_dir / "runs"
        if self.results_dir is None:
            self.results_dir = self.base_dir / "results"
        if self.summary_dir is None:
            self.summary_dir = self.base_dir / "summary"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "base_dir": str(self.base_dir),
            "tasks_dir": str(self.tasks_dir),
            "prompts_dir": str(self.prompts_dir),
            "tokens_dir": str(self.tokens_dir),
            "outputs_dir": str(self.outputs_dir),
            "runs_dir": str(self.runs_dir),
            "results_dir": str(self.results_dir),
            "summary_dir": str(self.summary_dir),
            "instances_per_family": self.instances_per_family,
            "seed": self.seed,
            "token_tolerance": self.token_tolerance,
            "tokenizer_model": self.tokenizer_model,
            "models": self.models,
            "backend": self.backend,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "max_tokens": self.max_tokens,
            "max_context_tokens": self.max_context_tokens,
            "request_delay": self.request_delay,
            "skip_existing": self.skip_existing,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """Create from dictionary."""
        return cls(**d)


class Pipeline:
    """Main pipeline orchestrator."""

    STAGE_NAMES = {
        1: "Dataset & Task Specification",
        2: "Prompt Construction",
        3: "Token Accounting & Matching",
        4: "Model Execution",
        5: "Automatic Evaluation",
        6: "Aggregation & Reporting",
    }

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.logger = self._setup_logging()
        self._ensure_directories()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger("metaglyph")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            logger.addHandler(handler)

        return logger

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        dirs = [
            self.config.tasks_dir,
            self.config.prompts_dir,
            self.config.tokens_dir,
            self.config.outputs_dir,
            self.config.runs_dir,
            self.config.results_dir,
            self.config.summary_dir,
        ]
        for d in dirs:
            ensure_dir(d)

    def run_all(self) -> dict[str, Any]:
        """Run the complete pipeline."""
        self.logger.info("Starting MetaGlyph Pipeline")
        start_time = time.time()

        results = {}

        for stage in range(1, 7):
            stage_result = self.run_stage(stage)
            results[f"stage_{stage}"] = stage_result

        elapsed = time.time() - start_time
        self.logger.info(f"Pipeline completed in {elapsed:.1f}s")

        # Save pipeline run summary
        results["config"] = self.config.to_dict()
        results["elapsed_seconds"] = elapsed
        save_json(results, self.config.summary_dir / "pipeline_run.json")

        return results

    def run_stage(self, stage: int) -> dict[str, Any]:
        """Run a specific pipeline stage."""
        stage_name = self.STAGE_NAMES.get(stage, f"Stage {stage}")
        self.logger.info(f"Running Stage {stage}: {stage_name}")
        start_time = time.time()

        try:
            if stage == 1:
                result = self._run_stage1()
            elif stage == 2:
                result = self._run_stage2()
            elif stage == 3:
                result = self._run_stage3()
            elif stage == 4:
                result = self._run_stage4()
            elif stage == 5:
                result = self._run_stage5()
            elif stage == 6:
                result = self._run_stage6()
            else:
                raise ValueError(f"Unknown stage: {stage}")

            elapsed = time.time() - start_time
            self.logger.info(f"Stage {stage} completed in {elapsed:.1f}s")

            return {
                "success": True,
                "elapsed_seconds": elapsed,
                "result": result,
            }

        except Exception as e:
            self.logger.error(f"Stage {stage} failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def _run_stage1(self) -> dict[str, Any]:
        """Stage 1: Dataset & Task Specification."""
        generator = DatasetGenerator(
            output_dir=self.config.tasks_dir,
            seed=self.config.seed,
        )

        instances = generator.generate_all(
            instances_per_family=self.config.instances_per_family
        )

        return {
            "families": list(instances.keys()),
            "instances_per_family": {
                family: len(insts) for family, insts in instances.items()
            },
        }

    def _run_stage2(self) -> dict[str, Any]:
        """Stage 2: Prompt Construction."""
        constructor = PromptConstructor(
            tasks_dir=self.config.tasks_dir,
            output_dir=self.config.prompts_dir,
        )

        prompts = constructor.construct_all()

        return {
            "families": list(prompts.keys()),
            "prompts_per_family": {
                family: len(prompt_list) for family, prompt_list in prompts.items()
            },
        }

    def _run_stage3(self) -> dict[str, Any]:
        """Stage 3: Token Accounting & Matching."""
        results = {}

        for model in self.config.models:
            matcher = TokenMatcher(
                prompts_dir=self.config.prompts_dir,
                tasks_dir=self.config.tasks_dir,
                output_dir=self.config.tokens_dir,
                model_name=model if self.config.tokenizer_model == "model" else self.config.tokenizer_model,
                tolerance=self.config.token_tolerance,
            )

            counts = matcher.validate_all()

            # Get compression stats
            compression_stats = []
            for family in counts.keys():
                stats = matcher.get_compression_stats(family)
                compression_stats.append(stats)

            results[model] = {
                "validated_prompts": sum(len(c) for c in counts.values()),
                "compression_stats": compression_stats,
            }

        return results

    def _run_stage4(self) -> dict[str, Any]:
        """Stage 4: Model Execution."""
        results = {}

        exec_config = ExecutionConfig(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
            max_tokens=self.config.max_tokens,
            max_context_tokens=self.config.max_context_tokens,
            seed=self.config.seed,
            request_delay=self.config.request_delay,
        )

        for model in self.config.models:
            executor = ModelExecutor(
                prompts_dir=self.config.prompts_dir,
                tasks_dir=self.config.tasks_dir,
                outputs_dir=self.config.outputs_dir,
                runs_dir=self.config.runs_dir,
                model_name=model,
                backend=self.config.backend,
                config=exec_config,
            )

            # Check availability
            if not executor.check_availability():
                self.logger.warning(f"Model {model} not available, skipping")
                results[model] = {"available": False, "executed": 0}
                continue

            # Execute
            exec_results = executor.execute_all(skip_existing=self.config.skip_existing)

            total_executed = sum(len(r) for r in exec_results.values())
            successful = sum(
                sum(1 for x in r if x.success)
                for r in exec_results.values()
            )

            results[model] = {
                "available": True,
                "executed": total_executed,
                "successful": successful,
            }

        return results

    def _run_stage5(self) -> dict[str, Any]:
        """Stage 5: Automatic Evaluation."""
        results = {}

        for model in self.config.models:
            evaluator = Evaluator(
                outputs_dir=self.config.outputs_dir,
                tasks_dir=self.config.tasks_dir,
                prompts_dir=self.config.prompts_dir,
                results_dir=self.config.results_dir,
                model_name=model,
            )

            eval_results = evaluator.evaluate_all()

            total_evaluated = sum(len(r) for r in eval_results.values())
            total_passed = sum(
                sum(1 for x in r if x.overall_pass)
                for r in eval_results.values()
            )

            results[model] = {
                "evaluated": total_evaluated,
                "passed": total_passed,
                "pass_rate": total_passed / total_evaluated if total_evaluated > 0 else 0,
            }

        return results

    def _run_stage6(self) -> dict[str, Any]:
        """Stage 6: Aggregation & Reporting."""
        aggregator = Aggregator(
            results_dir=self.config.results_dir,
            tokens_dir=self.config.tokens_dir,
            summary_dir=self.config.summary_dir,
            models=self.config.models,
            prompts_dir=self.config.prompts_dir,
            tokenizer_model=self.config.tokenizer_model,
        )

        summary = aggregator.aggregate_all()

        return {
            "summary_generated": True,
            "models_aggregated": self.config.models,
        }


def run_pipeline(
    base_dir: str = ".",
    models: list[str] = None,
    instances_per_family: int = 50,
    **kwargs,
) -> dict[str, Any]:
    """
    Convenience function to run the pipeline.

    Args:
        base_dir: Base directory for all artifacts
        models: List of model names to run
        instances_per_family: Number of task instances per family
        **kwargs: Additional config options

    Returns:
        Pipeline run results
    """
    config = PipelineConfig(
        base_dir=Path(base_dir),
        models=models or ["llama3.2"],
        instances_per_family=instances_per_family,
        **kwargs,
    )

    pipeline = Pipeline(config)
    return pipeline.run_all()
