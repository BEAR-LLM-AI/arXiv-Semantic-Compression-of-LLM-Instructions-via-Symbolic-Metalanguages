# MetaGlyph — Symbolic Metalanguages for LLM Prompting

This repository contains the research artifacts, datasets, and evaluation framework for studying **symbolic metalanguages for large language model (LLM) prompting**. The goal of this project is to evaluate whether mathematical and logical operators can be used to **semantically compress instruction language**, independent of context compression or learned prompt optimization.

The work is designed to be **fully automatic, reproducible, and long-context aware**, using **free-tier models via OpenRouter API**.

---

## Quick start

### 1. Setup

```bash
# Clone and install dependencies
git clone https://github.com/your-repo/metaglyph.git
cd metaglyph
pip install -r requirements.txt
```

### 2. Configure API key

Create a `.env` file with your OpenRouter API key:

```
OPENROUTER_API_KEY=your_api_key_here
```

Get a free API key at [openrouter.ai](https://openrouter.ai).

### 3. Create task instances

```bash
# Create 50 instances per task family (duplicates instance_001)
python create_instances.py --instances 50
```

### 4. Run experiments

```bash
# Run full pipeline (stages 3-6: tokens, execution, evaluation, aggregation)
python run_pipeline.py --stage 3-6

# Run only model execution (Stage 4)
python run_pipeline.py --stage 4

# Re-run evaluation and reporting only (Stage 5-6)
python run_pipeline.py --stage 5,6
```

---

## Models

All experiments use models via OpenRouter API:

| Model | OpenRouter ID | Size | Purpose |
|-------|---------------|------|---------|
| Llama 3.2 3B Instruct | `meta-llama/llama-3.2-3b-instruct:free` | 3B | Small dense baseline |
| Gemma 3 12B | `google/gemma-3-12b-it:free` | 12B | Mid-size instruction follower |

**Note:** Requires OpenRouter API credits. Free-tier models are also supported but have daily rate limits.

### Execution parameters

All models run with identical settings:
- `temperature: 0` (deterministic)
- `top_p: 1.0`
- `frequency_penalty: 0`
- `presence_penalty: 0`
- `max_tokens: 2048`

---

## Pipeline architecture

The pipeline has **six stages**, executed in order:

```
Stage 1: Dataset & Task Specification
    ↓
Stage 2: Prompt Construction (NL / MG / CTRL)
    ↓
Stage 3: Token Accounting & Matching
    ↓
Stage 4: Model Execution ← Only stage using LLMs
    ↓
Stage 5: Automatic Evaluation
    ↓
Stage 6: Aggregation & Reporting
```

### Stage outputs

| Stage | Artifacts |
|-------|-----------|
| 1 | `tasks/<family>/*.{input,gold,constraints,meta}` |
| 2 | `prompts/<family>/*.txt` |
| 3 | `tokens/<model>/*.json` |
| 4 | `outputs/<model>/*.txt`, `runs/<model>/*.meta` |
| 5 | `results/<model>/*.json` |
| 6 | `summary/tables/*.csv`, `summary/figures/*.pdf` |

---

## Project motivation

Natural-language prompts act as *instruction languages*, specifying how inputs should be selected, transformed, or constrained. While effective, natural language is verbose and ambiguous as a control channel. This project investigates whether **symbolic operators already internalized during model pretraining** (e.g., ∈, ¬, ∩, ⇒) can function as compact, reliable instruction-semantic primitives.

Unlike prompt compression systems that prune context, or constructed prompt languages that rely on system-level decoding schemes, this work focuses on **semantic compression of the instruction language itself**, under strict token control.

---

## Task families

Four task families, each testing different operator semantics:

| Family | Operators | Description |
|--------|-----------|-------------|
| Selection & Classification | ∈, ∉, ¬, ∩, ∪ | Select items based on set membership |
| Structured Extraction | ∈, →, ↦, \| | Extract fields from documents |
| Constraint Composition | ∩, ∪, ¬, ⊆, ∀, ∃ | Apply composed constraints |
| Conditional Transformation | ⇒, ∘, \|, → | Transform based on rules |

---

## Token compression results

MetaGlyph achieves significant token reduction compared to natural language instructions:

| Task Family | NL Tokens | MG Tokens | CTRL Tokens | Reduction |
|-------------|-----------|-----------|-------------|-----------|
| Selection & Classification | 215 | 41 | 41 | **80.9%** |
| Structured Extraction | 176 | 52 | 52 | **70.5%** |
| Constraint Composition | 134 | 48 | 48 | **64.2%** |
| Conditional Transformation | 164 | 62 | 62 | **62.2%** |

**Average token reduction: 69.5%**

The CTRL condition uses the same token count as MG but with semantically broken symbols, isolating the effect of operator semantics from mere token compression.

---

## Experimental design

Each task instance is evaluated under three instruction conditions:

1. **NL** — verbose natural-language instruction
2. **MG** — compact MetaGlyph symbolic instruction
3. **CTRL** — symbol-shaped control (same structure, broken semantics)

Instruction token counts are matched across conditions to isolate **semantic effects** from length/formatting effects.

---

## Symbolic operator inventory

MetaGlyph uses high-frequency mathematical and logical operators:

| Category | Operators |
|----------|-----------|
| Transformation | `→`, `⇒`, `∘`, `↦` |
| Set/constraints | `∈`, `∉`, `⊆`, `∩`, `∪` |
| Logical | `¬`, `∀`, `∃` |
| Scope | `\|` |

---

## CLI reference

```bash
# Full pipeline
python run_pipeline.py

# Specific stages
python run_pipeline.py --stage 1        # Dataset generation only
python run_pipeline.py --stage 1-3      # Stages 1 through 3
python run_pipeline.py --stage 4,5,6    # Execution + evaluation

# Configuration
python run_pipeline.py --instances 50   # 50 instances per family
python run_pipeline.py --models llama-3.2-3b,qwen3-8b
python run_pipeline.py --backend openrouter
python run_pipeline.py --config custom.json

# With custom config file
python run_pipeline.py --config my_config.json
```

---

## Repository structure

```
.
├── src/
│   ├── pipeline.py              # Main orchestrator
│   ├── stages/
│   │   ├── stage1_dataset.py    # Task generation
│   │   ├── stage2_prompts.py    # Prompt construction
│   │   ├── stage3_tokens.py     # Token matching
│   │   ├── stage4_execution.py  # Model execution
│   │   ├── stage5_evaluation.py # Scoring
│   │   └── stage6_aggregation.py# Reporting
│   └── utils/
│       ├── operators.py         # Operator definitions
│       ├── tokenizers.py        # Model tokenizers
│       └── io_utils.py          # File I/O
├── tasks/                       # Task instances (generated)
├── prompts/                     # Prompt files
├── outputs/                     # Model outputs
├── results/                     # Evaluation results
├── summary/                     # Tables and figures
├── config.json                  # Default configuration
├── requirements.txt
├── run_pipeline.py              # CLI entry point
└── .env                         # API keys (not committed)
```

---

## Reproducibility

- All experiments use **free-tier open-weight models** via OpenRouter
- Model IDs, decoding parameters, and seeds are fixed
- Results can be regenerated end-to-end
- No manual inspection required for scoring

---

## Scope and limitations

This repository focuses on **instruction semantics**, not system performance. It does **not** claim or measure:

- Latency improvements
- Memory usage
- Attention complexity
- Throughput or cost savings

The experiments are single-turn and do not evaluate multi-turn dialogue.

---

## Citation

```bibtex
@article{metaglyph2025,
  title={Semantic Compression of LLM Instructions via Symbolic Metalanguages},
  author={Ernst van Gassen},
  journal={arXiv preprint},
  year={2025}
}
```

---

## License

This repository is released for research and academic use. See the `LICENSE` file for details.
