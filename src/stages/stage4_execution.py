"""Stage 4: Model Execution.

This module runs validated prompts through language models.
It is the ONLY stage where models are actually invoked.

Execution settings:
- Deterministic decoding (temperature=0)
- Fixed context length
- Fixed output token limit
- Same settings across NL/MG/CTRL conditions

Supported backends:
- Ollama (local, primary)
- OpenAI API (optional validation)
- Anthropic API (optional validation)

Artifacts produced:
- outputs/<model>/<prompt_id>.txt  - Raw model output
- runs/<model>/<prompt_id>.meta    - Runtime metadata
"""

import time
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional
from abc import ABC, abstractmethod
import hashlib

from ..utils.io_utils import load_json, load_text, save_json, save_text, ensure_dir, list_files


@dataclass
class ExecutionConfig:
    """Configuration for model execution."""
    temperature: float = 0.0  # Deterministic
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 781
    max_context_tokens: int = 8192
    timeout_seconds: int = 120
    seed: Optional[int] = 42  # For reproducibility where supported
    request_delay: float = 0.5  # Delay between requests (seconds) to respect rate limits


@dataclass
class ExecutionResult:
    """Result of a single model execution."""
    prompt_id: str
    model: str
    raw_output: str
    success: bool
    error_message: Optional[str]
    execution_time_ms: int
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    config: dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunMetadata:
    """Metadata for a model run."""
    prompt_id: str
    model: str
    model_version: Optional[str]
    timestamp: str
    execution_time_ms: int
    config: dict
    prompt_hash: str
    success: bool
    error: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


class ModelBackend(ABC):
    """Abstract base class for model backends."""

    def __init__(self, model_name: str, config: ExecutionConfig):
        self.model_name = model_name
        self.config = config

    @abstractmethod
    def generate(self, prompt: str) -> ExecutionResult:
        """Generate a response for the given prompt."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available."""
        pass

    def get_model_version(self) -> Optional[str]:
        """Get the specific model version if available."""
        return None


class OllamaBackend(ModelBackend):
    """Backend for local Ollama models."""

    def __init__(self, model_name: str, config: ExecutionConfig, base_url: str = "http://localhost:11434"):
        super().__init__(model_name, config)
        self.base_url = base_url

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(m.get("name", "").startswith(self.model_name) for m in models)
            return False
        except Exception:
            return False

    def generate(self, prompt: str) -> ExecutionResult:
        """Generate using Ollama API."""
        import requests

        start_time = time.time()
        prompt_id = hashlib.md5(prompt.encode()).hexdigest()[:8]

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                        "repeat_penalty": 1.0 + self.config.frequency_penalty,  # Ollama equivalent
                        "num_predict": self.config.max_tokens,
                        "num_ctx": self.config.max_context_tokens,
                        "seed": self.config.seed,
                    },
                },
                timeout=self.config.timeout_seconds,
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                result = response.json()
                return ExecutionResult(
                    prompt_id=prompt_id,
                    model=self.model_name,
                    raw_output=result.get("response", ""),
                    success=True,
                    error_message=None,
                    execution_time_ms=elapsed_ms,
                    input_tokens=result.get("prompt_eval_count"),
                    output_tokens=result.get("eval_count"),
                    config=asdict(self.config),
                )
            else:
                return ExecutionResult(
                    prompt_id=prompt_id,
                    model=self.model_name,
                    raw_output="",
                    success=False,
                    error_message=f"HTTP {response.status_code}: {response.text}",
                    execution_time_ms=elapsed_ms,
                    input_tokens=None,
                    output_tokens=None,
                    config=asdict(self.config),
                )

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                prompt_id=prompt_id,
                model=self.model_name,
                raw_output="",
                success=False,
                error_message=str(e),
                execution_time_ms=elapsed_ms,
                input_tokens=None,
                output_tokens=None,
                config=asdict(self.config),
            )

    def get_model_version(self) -> Optional[str]:
        """Get Ollama model version."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                for m in models:
                    if m.get("name", "").startswith(self.model_name):
                        return m.get("digest", "")[:12]
        except Exception:
            pass
        return None


class OpenAIBackend(ModelBackend):
    """Backend for OpenAI API models."""

    def __init__(self, model_name: str, config: ExecutionConfig, api_key: Optional[str] = None):
        super().__init__(model_name, config)
        self.api_key = api_key

    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        if not self.api_key:
            import os
            self.api_key = os.environ.get("OPENAI_API_KEY")
        return bool(self.api_key)

    def generate(self, prompt: str) -> ExecutionResult:
        """Generate using OpenAI API."""
        try:
            from openai import OpenAI
        except ImportError:
            return ExecutionResult(
                prompt_id="",
                model=self.model_name,
                raw_output="",
                success=False,
                error_message="openai package not installed",
                execution_time_ms=0,
                input_tokens=None,
                output_tokens=None,
                config=asdict(self.config),
            )

        start_time = time.time()
        prompt_id = hashlib.md5(prompt.encode()).hexdigest()[:8]

        try:
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                max_tokens=self.config.max_tokens,
                seed=self.config.seed,
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            return ExecutionResult(
                prompt_id=prompt_id,
                model=self.model_name,
                raw_output=response.choices[0].message.content or "",
                success=True,
                error_message=None,
                execution_time_ms=elapsed_ms,
                input_tokens=response.usage.prompt_tokens if response.usage else None,
                output_tokens=response.usage.completion_tokens if response.usage else None,
                config=asdict(self.config),
            )

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                prompt_id=prompt_id,
                model=self.model_name,
                raw_output="",
                success=False,
                error_message=str(e),
                execution_time_ms=elapsed_ms,
                input_tokens=None,
                output_tokens=None,
                config=asdict(self.config),
            )


class AnthropicBackend(ModelBackend):
    """Backend for Anthropic API models."""

    def __init__(self, model_name: str, config: ExecutionConfig, api_key: Optional[str] = None):
        super().__init__(model_name, config)
        self.api_key = api_key

    def is_available(self) -> bool:
        """Check if Anthropic API is available."""
        if not self.api_key:
            import os
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        return bool(self.api_key)

    def generate(self, prompt: str) -> ExecutionResult:
        """Generate using Anthropic API."""
        try:
            import anthropic
        except ImportError:
            return ExecutionResult(
                prompt_id="",
                model=self.model_name,
                raw_output="",
                success=False,
                error_message="anthropic package not installed",
                execution_time_ms=0,
                input_tokens=None,
                output_tokens=None,
                config=asdict(self.config),
            )

        start_time = time.time()
        prompt_id = hashlib.md5(prompt.encode()).hexdigest()[:8]

        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                messages=[{"role": "user", "content": prompt}],
            )

            elapsed_ms = int((time.time() - start_time) * 1000)
            output_text = response.content[0].text if response.content else ""

            return ExecutionResult(
                prompt_id=prompt_id,
                model=self.model_name,
                raw_output=output_text,
                success=True,
                error_message=None,
                execution_time_ms=elapsed_ms,
                input_tokens=response.usage.input_tokens if response.usage else None,
                output_tokens=response.usage.output_tokens if response.usage else None,
                config=asdict(self.config),
            )

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                prompt_id=prompt_id,
                model=self.model_name,
                raw_output="",
                success=False,
                error_message=str(e),
                execution_time_ms=elapsed_ms,
                input_tokens=None,
                output_tokens=None,
                config=asdict(self.config),
            )


class OpenRouterBackend(ModelBackend):
    """Backend for OpenRouter API (supports many models via unified API)."""

    # Model ID mapping for convenience (paid versions - no :free suffix)
    MODEL_IDS = {
        "llama-3.2-3b": "meta-llama/llama-3.2-3b-instruct",
        "qwen3-8b": "qwen/qwen3-8b",
        "gemma-3-12b": "google/gemma-3-12b-it",
        "olmo-2-32b": "allenai/olmo-2-0325-32b-instruct",
        "kimi-k2": "moonshotai/kimi-k2",
    }

    def __init__(
        self,
        model_name: str,
        config: ExecutionConfig,
        api_key: Optional[str] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
    ):
        super().__init__(model_name, config)
        self.api_key = api_key
        self.site_url = site_url
        self.site_name = site_name
        self.base_url = "https://openrouter.ai/api/v1"

        # Resolve model ID if short name provided
        self.model_id = self.MODEL_IDS.get(model_name, model_name)

    def is_available(self) -> bool:
        """Check if OpenRouter API is available."""
        if not self.api_key:
            import os
            self.api_key = os.environ.get("OPENROUTER_API_KEY")
            self.site_url = self.site_url or os.environ.get("OPENROUTER_SITE_URL", "")
            self.site_name = self.site_name or os.environ.get("OPENROUTER_SITE_NAME", "MetaGlyph")
        return bool(self.api_key)

    def generate(self, prompt: str) -> ExecutionResult:
        """Generate using OpenRouter API."""
        import requests

        start_time = time.time()
        prompt_id = hashlib.md5(prompt.encode()).hexdigest()[:8]

        if not self.api_key:
            import os
            self.api_key = os.environ.get("OPENROUTER_API_KEY")

        if not self.api_key:
            return ExecutionResult(
                prompt_id=prompt_id,
                model=self.model_id,
                raw_output="",
                success=False,
                error_message="OPENROUTER_API_KEY not set",
                execution_time_ms=0,
                input_tokens=None,
                output_tokens=None,
                config=asdict(self.config),
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "frequency_penalty": self.config.frequency_penalty,
                    "presence_penalty": self.config.presence_penalty,
                    "max_tokens": self.config.max_tokens,
                },
                timeout=self.config.timeout_seconds,
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                result = response.json()
                choices = result.get("choices", [])
                output_text = ""
                if choices:
                    message = choices[0].get("message", {})
                    output_text = message.get("content", "")

                usage = result.get("usage", {})

                return ExecutionResult(
                    prompt_id=prompt_id,
                    model=self.model_id,
                    raw_output=output_text,
                    success=True,
                    error_message=None,
                    execution_time_ms=elapsed_ms,
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                    config=asdict(self.config),
                )
            else:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = f"{error_msg}: {error_data['error'].get('message', response.text)}"
                except Exception:
                    error_msg = f"{error_msg}: {response.text}"

                return ExecutionResult(
                    prompt_id=prompt_id,
                    model=self.model_id,
                    raw_output="",
                    success=False,
                    error_message=error_msg,
                    execution_time_ms=elapsed_ms,
                    input_tokens=None,
                    output_tokens=None,
                    config=asdict(self.config),
                )

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                prompt_id=prompt_id,
                model=self.model_id,
                raw_output="",
                success=False,
                error_message=str(e),
                execution_time_ms=elapsed_ms,
                input_tokens=None,
                output_tokens=None,
                config=asdict(self.config),
            )

    def get_model_version(self) -> Optional[str]:
        """Return the model ID as version."""
        return self.model_id


class ModelExecutor:
    """Main class for executing prompts against models."""

    BACKENDS = {
        "ollama": OllamaBackend,
        "openai": OpenAIBackend,
        "anthropic": AnthropicBackend,
        "openrouter": OpenRouterBackend,
    }

    # Instruction conditions
    CONDITIONS = ["NL", "MG", "CTRL"]

    # Output format strings per family
    OUTPUT_FORMATS = {
        "1_selection_classification": "Return your answer as a JSON array of strings.",
        "2_structured_extraction": "Return your answer as a JSON object.",
        "3_constraint_composition": "Return your answer as a JSON object with 'selected_ids' and 'count'.",
        "4_conditional_transformation": "Return your answer as a JSON array of transformed objects.",
    }

    def __init__(
        self,
        prompts_dir: Path | str,
        tasks_dir: Path | str,
        outputs_dir: Path | str,
        runs_dir: Path | str,
        model_name: str,
        backend: str = "ollama",
        config: Optional[ExecutionConfig] = None,
    ):
        self.prompts_dir = Path(prompts_dir)
        self.tasks_dir = Path(tasks_dir)
        self.outputs_dir = Path(outputs_dir)
        self.runs_dir = Path(runs_dir)
        self.model_name = model_name
        self.config = config or ExecutionConfig()

        # Initialize backend
        backend_class = self.BACKENDS.get(backend, OllamaBackend)
        self.backend = backend_class(model_name, self.config)

    def check_availability(self) -> bool:
        """Check if the model backend is available."""
        return self.backend.is_available()

    def execute_all(self, skip_existing: bool = True) -> dict[str, list[ExecutionResult]]:
        """Execute all prompts."""
        all_results = {}

        # Get list of families
        families = [d.name for d in self.prompts_dir.iterdir() if d.is_dir()]
        total_families = len(families)

        print(f"\n[{self.model_name}] Starting execution for {total_families} families...")

        for family_idx, family_name in enumerate(sorted(families), 1):
            print(f"\n[{self.model_name}] Family {family_idx}/{total_families}: {family_name}")
            results = self._execute_family(family_name, skip_existing)
            all_results[family_name] = results

            executed = len(results)
            successful = sum(1 for r in results if r.success)
            print(f"[{self.model_name}] Completed: {executed} executed, {successful} successful")

        return all_results

    def execute_family(self, family_name: str, skip_existing: bool = True) -> list[ExecutionResult]:
        """Execute prompts for a specific family."""
        return self._execute_family(family_name, skip_existing)

    def _execute_family(self, family_name: str, skip_existing: bool) -> list[ExecutionResult]:
        """Internal method to execute family prompts.

        Combines instruction files (NL.txt, MG.txt, CTRL.txt) with input files
        (instance_XXX.input) to create full prompts at runtime.
        """
        prompts_family_dir = self.prompts_dir / family_name
        tasks_family_dir = self.tasks_dir / family_name
        output_dir = self.outputs_dir / self.model_name / family_name
        runs_dir_model = self.runs_dir / self.model_name / family_name

        ensure_dir(output_dir)
        ensure_dir(runs_dir_model)

        results = []

        # Load instruction files (NL.txt, MG.txt, CTRL.txt)
        instructions = {}
        for condition in self.CONDITIONS:
            instruction_path = prompts_family_dir / f"{condition}.txt"
            if instruction_path.exists():
                instructions[condition] = load_text(instruction_path)

        if not instructions:
            return results

        # Find all input files (instance_XXX.input)
        input_files = sorted(list_files(tasks_family_dir, "*.input"))

        # Get output format for this family
        output_format = self.OUTPUT_FORMATS.get(family_name, "Return your answer as JSON.")

        # Calculate total requests for progress
        total_instances = len(input_files)
        total_conditions = len(instructions)
        total_requests = total_instances * total_conditions

        # Count existing (to skip)
        skipped = 0
        if skip_existing:
            for input_file in input_files:
                instance_id = input_file.stem.replace(".input", "")
                for condition in instructions.keys():
                    output_path = output_dir / f"{instance_id}_{condition}.txt"
                    if output_path.exists():
                        skipped += 1

        to_execute = total_requests - skipped
        if skipped > 0:
            print(f"  Skipping {skipped} existing, executing {to_execute} remaining...")

        executed = 0
        for input_file in input_files:
            instance_id = input_file.stem.replace(".input", "")
            input_text = load_text(input_file)

            # Run each condition (NL, MG, CTRL) for this instance
            for condition, instruction in instructions.items():
                prompt_id = f"{instance_id}_{condition}"
                output_path = output_dir / f"{prompt_id}.txt"

                # Skip if already processed
                if skip_existing and output_path.exists():
                    continue

                executed += 1
                # Progress indicator
                print(f"  [{executed}/{to_execute}] {prompt_id}...", end=" ", flush=True)

                # Assemble full prompt
                full_prompt = self._assemble_prompt(instruction, input_text, output_format)

                # Execute
                result = self._execute_single(prompt_id, full_prompt)
                results.append(result)

                # Show result
                if result.success:
                    print(f"OK ({result.execution_time_ms}ms)")
                else:
                    print(f"FAILED: {result.error_message[:50]}")

                # Save outputs
                self._save_result(result, output_dir, runs_dir_model)

                # Polite delay to respect rate limits
                if self.config.request_delay > 0:
                    time.sleep(self.config.request_delay)

        return results

    def _assemble_prompt(self, instruction: str, input_text: str, output_format: str) -> str:
        """Assemble full prompt from instruction, input, and output format."""
        return f"""### Instruction
{instruction}

### Input
{input_text}

### Output Format
{output_format}

### Response"""

    def _execute_single(self, prompt_id: str, prompt_text: str) -> ExecutionResult:
        """Execute a single prompt."""
        result = self.backend.generate(prompt_text)
        result.prompt_id = prompt_id
        return result

    def _save_result(self, result: ExecutionResult, output_dir: Path, runs_dir: Path) -> None:
        """Save execution result and metadata."""
        import datetime

        # Save raw output
        save_text(result.raw_output, output_dir / f"{result.prompt_id}.txt")

        # Create and save run metadata
        prompt_hash = hashlib.md5(result.raw_output.encode()).hexdigest()[:16]

        metadata = RunMetadata(
            prompt_id=result.prompt_id,
            model=result.model,
            model_version=self.backend.get_model_version(),
            timestamp=datetime.datetime.now().isoformat(),
            execution_time_ms=result.execution_time_ms,
            config=result.config,
            prompt_hash=prompt_hash,
            success=result.success,
            error=result.error_message,
        )

        save_json(metadata.to_dict(), runs_dir / f"{result.prompt_id}.meta")

    def execute_prompt(self, prompt_id: str, prompt_text: str) -> ExecutionResult:
        """Execute a single prompt directly."""
        return self._execute_single(prompt_id, prompt_text)


class BatchExecutor:
    """Utility for batch execution with progress tracking."""

    def __init__(self, executor: ModelExecutor):
        self.executor = executor

    def execute_with_progress(
        self,
        prompts: list[tuple[str, str]],  # (prompt_id, prompt_text)
        callback: Optional[callable] = None,
    ) -> list[ExecutionResult]:
        """
        Execute prompts with progress callback.

        callback(current, total, result) is called after each execution.
        """
        results = []
        total = len(prompts)

        for i, (prompt_id, prompt_text) in enumerate(prompts):
            result = self.executor.execute_prompt(prompt_id, prompt_text)
            results.append(result)

            if callback:
                callback(i + 1, total, result)

        return results
