"""Tokenizer utilities for different models.

This module provides tokenizer mappings and utilities for counting
tokens across different model families.

HuggingFace tokenizer IDs for the test models:
- Llama 3.2 3B: meta-llama/Llama-3.2-3B-Instruct
- Qwen3 4B: Qwen/Qwen3-4B
- Gemma 3 12B: google/gemma-3-12b-it
- OLMo 3 32B: allenai/OLMo-3-32B-Think
"""

from typing import Optional
import os


# Mapping from short model names to HuggingFace tokenizer IDs
TOKENIZER_MAP = {
    # Llama family
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/llama-3.2-3b-instruct:free": "meta-llama/Llama-3.2-3B-Instruct",

    # Qwen family
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen/qwen3-8b": "Qwen/Qwen3-8B",

    # Gemma family
    "gemma-3-12b": "google/gemma-3-12b-it",
    "google/gemma-3-12b-it:free": "google/gemma-3-12b-it",

    # OLMo family
    "olmo-2-32b": "allenai/OLMo-2-0325-32B-Instruct",
    "allenai/olmo-2-0325-32b-instruct": "allenai/OLMo-2-0325-32B-Instruct",

    # Kimi (MoE)
    "kimi-k2": "moonshotai/kimi-k2",
    "moonshotai/kimi-k2:free": "moonshotai/kimi-k2",
}


class ModelTokenizer:
    """Wrapper for model-specific tokenizers."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer_id = TOKENIZER_MAP.get(model_name, model_name)
        self._tokenizer = None
        self._load_attempted = False

    def _load_tokenizer(self):
        """Lazy load the tokenizer."""
        if self._load_attempted:
            return

        self._load_attempted = True

        try:
            from transformers import AutoTokenizer

            # Some models require authentication
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_id,
                token=hf_token,
                trust_remote_code=True,  # Some models need this
            )
        except Exception as e:
            print(f"Warning: Could not load tokenizer for {self.tokenizer_id}: {e}")
            self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        self._load_tokenizer()

        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))

        # Fallback: approximate with word count * 1.3
        return int(len(text.split()) * 1.3)

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text and return token strings."""
        self._load_tokenizer()

        if self._tokenizer is not None:
            token_ids = self._tokenizer.encode(text)
            return [self._tokenizer.decode([tid]) for tid in token_ids]

        # Fallback: simple whitespace split
        return text.split()

    def is_available(self) -> bool:
        """Check if the tokenizer loaded successfully."""
        self._load_tokenizer()
        return self._tokenizer is not None


def get_tokenizer(model_name: str) -> ModelTokenizer:
    """Get a tokenizer for the specified model."""
    return ModelTokenizer(model_name)


def count_tokens(text: str, model_name: str) -> int:
    """Count tokens in text for the specified model."""
    tokenizer = get_tokenizer(model_name)
    return tokenizer.count_tokens(text)


# Pre-defined tokenizers for quick access
TOKENIZERS = {
    name: ModelTokenizer(name) for name in TOKENIZER_MAP.keys()
}
