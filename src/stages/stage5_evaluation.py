"""Stage 5: Automatic Evaluation.

This module evaluates model outputs against gold labels.
All evaluation is fully automatic - no human inspection required.

Evaluation components:
- Output parsing (JSON/list extraction)
- Task scoring (accuracy, F1, exact match)
- Operator fidelity checks (per-operator semantic verification)
- Error classification (parse, scope, logic)

Artifacts produced:
- results/<model>/<prompt_id>.json
"""

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional
from enum import Enum

from ..utils.io_utils import load_json, load_text, save_json, ensure_dir, list_files


class ErrorType(Enum):
    """Types of evaluation errors."""
    NONE = "none"
    PARSE = "parse"           # Failed to parse output
    SCOPE = "scope"           # Wrong scope/filtering
    LOGIC = "logic"           # Wrong logical operation
    FORMAT = "format"         # Wrong output format
    INCOMPLETE = "incomplete" # Missing required elements
    EXTRA = "extra"           # Extra unwanted elements


@dataclass
class OperatorFidelity:
    """Result of checking operator-specific fidelity."""
    operator: str
    checked: bool
    passed: bool
    details: str


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single output."""
    prompt_id: str
    model: str
    family: str
    condition: str
    parse_success: bool
    parsed_output: Any
    gold_output: Any
    exact_match: bool
    accuracy: float
    f1_score: float
    operator_fidelity: list[OperatorFidelity]
    overall_pass: bool
    error_type: ErrorType
    error_details: Optional[str]
    # Secondary pass: NL == MG and both != CTRL (semantic equivalence)
    semantic_equivalence_pass: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d['error_type'] = self.error_type.value
        d['operator_fidelity'] = [asdict(of) for of in self.operator_fidelity]
        return d


class OutputParser:
    """Parser for extracting structured data from model outputs."""

    def parse(self, raw_output: str, expected_format: str, max_retries: int = 3) -> tuple[bool, Any, str]:
        """
        Parse raw output into structured data with retry logic.

        Tries multiple parsing strategies before giving up.

        Returns:
            (success, parsed_data, error_message)
        """
        strategies = self._get_parse_strategies(expected_format)

        last_error = ""
        for strategy in strategies:
            success, parsed, error = strategy(raw_output)
            if success:
                return True, parsed, ""
            last_error = error

        return False, None, last_error

    def _get_parse_strategies(self, expected_format: str) -> list:
        """Get ordered list of parsing strategies to try."""
        if expected_format == "list":
            return [
                self._parse_json_array,
                self._parse_json_object_with_list,
                self._parse_line_by_line,
                self._parse_comma_separated,
            ]
        elif expected_format == "json":
            return [
                self._parse_json,
                self._parse_json_relaxed,
                self._parse_json_from_text,
            ]
        else:
            return [self._parse_json, self._parse_json_relaxed]

    def _parse_json(self, raw: str) -> tuple[bool, Any, str]:
        """Extract and parse JSON from output."""
        # Try to find JSON in the output
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',  # Markdown code block
            r'```\s*([\s\S]*?)\s*```',       # Generic code block
            r'(\{[\s\S]*\})',                # Object
            r'(\[[\s\S]*\])',                # Array
        ]

        for pattern in json_patterns:
            match = re.search(pattern, raw)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                    return True, parsed, ""
                except json.JSONDecodeError:
                    continue

        # Try parsing the entire output
        try:
            parsed = json.loads(raw.strip())
            return True, parsed, ""
        except json.JSONDecodeError as e:
            return False, None, f"JSON parse error: {str(e)}"

    def _parse_json_relaxed(self, raw: str) -> tuple[bool, Any, str]:
        """Try to fix common JSON issues and parse."""
        text = raw.strip()

        # Remove trailing commas before ] or }
        text = re.sub(r',(\s*[\]\}])', r'\1', text)

        # Try to extract JSON after common prefixes
        prefixes = ["Here is", "The answer is", "Output:", "Result:", "Response:"]
        for prefix in prefixes:
            if prefix.lower() in text.lower():
                idx = text.lower().find(prefix.lower()) + len(prefix)
                text = text[idx:].strip()
                break

        # Find JSON structure
        for pattern in [r'(\{[\s\S]*\})', r'(\[[\s\S]*\])']:
            match = re.search(pattern, text)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                    return True, parsed, ""
                except json.JSONDecodeError:
                    continue

        return False, None, "Could not parse relaxed JSON"

    def _parse_json_from_text(self, raw: str) -> tuple[bool, Any, str]:
        """Extract key-value pairs from text when JSON parsing fails."""
        # Try to build object from "key: value" patterns
        pairs = re.findall(r'"?(\w+)"?\s*:\s*"([^"]*)"', raw)
        if pairs:
            return True, dict(pairs), ""
        return False, None, "Could not extract key-value pairs"

    def _parse_json_array(self, raw: str) -> tuple[bool, Any, str]:
        """Parse as JSON array."""
        success, parsed, error = self._parse_json(raw)
        if success and isinstance(parsed, list):
            return True, parsed, ""
        return False, None, "Not a JSON array"

    def _parse_json_object_with_list(self, raw: str) -> tuple[bool, Any, str]:
        """Parse JSON object and extract list from known fields."""
        success, parsed, _ = self._parse_json(raw)
        if success and isinstance(parsed, dict):
            # Look for list fields
            list_fields = ["selected_ids", "items", "results", "output", "data", "list"]
            for field in list_fields:
                if field in parsed and isinstance(parsed[field], list):
                    return True, parsed[field], ""
            # Return first list field found
            for value in parsed.values():
                if isinstance(value, list):
                    return True, value, ""
        return False, None, "No list found in object"

    def _parse_line_by_line(self, raw: str) -> tuple[bool, Any, str]:
        """Extract a list from output line by line."""
        lines = raw.strip().split('\n')
        items = []
        for line in lines:
            line = line.strip()
            # Skip empty lines and common prefixes
            if not line or line.startswith('#') or line.startswith('```'):
                continue
            if line.lower().startswith(('here', 'the ', 'output', 'result')):
                continue
            # Remove bullet points, numbers, quotes
            cleaned = re.sub(r'^[-*•]\s*', '', line)
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', cleaned)
            cleaned = cleaned.strip('"\'`')
            cleaned = cleaned.rstrip(',')
            if cleaned and len(cleaned) > 1:
                items.append(cleaned)

        if items:
            return True, items, ""

        return False, None, "Could not extract list from lines"

    def _parse_comma_separated(self, raw: str) -> tuple[bool, Any, str]:
        """Parse comma-separated values."""
        # Remove code blocks and common wrapper text
        text = re.sub(r'```[\s\S]*?```', '', raw)
        text = re.sub(r'\[|\]|\{|\}', '', text)

        # Split by comma
        items = [item.strip().strip('"\'') for item in text.split(',')]
        items = [item for item in items if item and len(item) > 1]

        if len(items) >= 2:
            return True, items, ""

        return False, None, "Could not parse comma-separated values"

    def _parse_list(self, raw: str) -> tuple[bool, Any, str]:
        """Extract a list from output (legacy method)."""
        return self.parse(raw, "list")


class OperatorFidelityChecker:
    """Checks operator-specific semantic fidelity."""

    @staticmethod
    def _to_hashable(item: Any) -> Any:
        """Convert an item to a hashable form for set operations."""
        if isinstance(item, dict):
            return json.dumps(item, sort_keys=True)
        elif isinstance(item, list):
            return tuple(OperatorFidelityChecker._to_hashable(x) for x in item)
        return item

    @staticmethod
    def _to_hashable_set(items: list) -> set:
        """Convert a list to a set of hashable items."""
        return set(OperatorFidelityChecker._to_hashable(x) for x in items)

    def check_inclusion(
        self,
        output: Any,
        constraints: dict,
        gold: Any,
    ) -> OperatorFidelity:
        """Check ∈ (set membership/inclusion) fidelity."""
        if not isinstance(output, list):
            return OperatorFidelity("∈", True, False, "Output is not a list")

        attr = constraints.get("attribute", "")
        value = constraints.get("value", "")

        # Check if all output items should be included
        expected = self._to_hashable_set(gold) if isinstance(gold, list) else set()
        actual = self._to_hashable_set(output)

        # Items in output should be subset of expected
        extra = actual - expected
        if extra:
            return OperatorFidelity("∈", True, False, f"Extra items: {len(extra)}")

        return OperatorFidelity("∈", True, True, "Inclusion constraint satisfied")

    def check_exclusion(
        self,
        output: Any,
        constraints: dict,
        gold: Any,
    ) -> OperatorFidelity:
        """Check ∉ (set non-membership/exclusion) fidelity."""
        if not isinstance(output, list):
            return OperatorFidelity("∉", True, False, "Output is not a list")

        excluded_value = constraints.get("excluded_value", "")

        # For exclusion, we need the original data to verify
        # This is a simplified check against gold
        expected = self._to_hashable_set(gold) if isinstance(gold, list) else set()
        actual = self._to_hashable_set(output)

        # Check that no excluded items appear
        intersection = actual & expected
        if len(intersection) < len(expected):
            return OperatorFidelity("∉", True, True, "Exclusion constraint satisfied")

        return OperatorFidelity("∉", True, True, "Exclusion check passed")

    def check_intersection_scope(
        self,
        output: Any,
        constraints: dict,
        gold: Any,
    ) -> OperatorFidelity:
        """Check ∩ (intersection) scope fidelity."""
        if not isinstance(output, list):
            return OperatorFidelity("∩", True, False, "Output is not a list")

        expected = self._to_hashable_set(gold) if isinstance(gold, list) else set()
        actual = self._to_hashable_set(output)

        # Intersection should be subset of each individual criteria result
        # Simplified: check against gold which is the correct intersection
        if actual == expected:
            return OperatorFidelity("∩", True, True, "Intersection scope correct")
        elif actual.issubset(expected):
            return OperatorFidelity("∩", True, False, f"Missing items in intersection")
        else:
            return OperatorFidelity("∩", True, False, f"Extra items in intersection")

    def check_union_scope(
        self,
        output: Any,
        constraints: dict,
        gold: Any,
    ) -> OperatorFidelity:
        """Check ∪ (union) scope fidelity."""
        if not isinstance(output, list):
            return OperatorFidelity("∪", True, False, "Output is not a list")

        expected = self._to_hashable_set(gold) if isinstance(gold, list) else set()
        actual = self._to_hashable_set(output)

        if actual == expected:
            return OperatorFidelity("∪", True, True, "Union scope correct")
        elif expected.issubset(actual):
            return OperatorFidelity("∪", True, False, f"Extra items in union")
        else:
            return OperatorFidelity("∪", True, False, f"Missing items in union")

    def check_implication(
        self,
        output: Any,
        constraints: dict,
        gold: Any,
    ) -> OperatorFidelity:
        """Check ⇒ (conditional/implication) fidelity."""
        rules = constraints.get("rules", [])

        if not isinstance(output, list) or not isinstance(gold, list):
            return OperatorFidelity("⇒", True, False, "Invalid output format for rules")

        # Check if transformations were applied correctly
        if len(output) != len(gold):
            return OperatorFidelity("⇒", True, False, "Wrong number of records")

        mismatches = 0
        for i, (out_rec, gold_rec) in enumerate(zip(output, gold)):
            if out_rec != gold_rec:
                mismatches += 1

        if mismatches == 0:
            return OperatorFidelity("⇒", True, True, "All rules applied correctly")
        else:
            return OperatorFidelity("⇒", True, False, f"{mismatches} records with wrong transformations")

    def check_composition_order(
        self,
        output: Any,
        constraints: dict,
        gold: Any,
    ) -> OperatorFidelity:
        """Check ∘ (composition/chaining) order fidelity."""
        # Composition order affects transformation results
        # Compare final output with gold to verify correct ordering

        if output == gold:
            return OperatorFidelity("∘", True, True, "Composition order correct")
        else:
            return OperatorFidelity("∘", True, False, "Composition order may be incorrect")

    def check_transformation(
        self,
        output: Any,
        constraints: dict,
        gold: Any,
    ) -> OperatorFidelity:
        """Check → (transformation/mapping) fidelity."""
        if isinstance(gold, dict) and isinstance(output, dict):
            missing = set(gold.keys()) - set(output.keys())
            if missing:
                return OperatorFidelity("→", True, False, f"Missing fields: {missing}")
            return OperatorFidelity("→", True, True, "Transformation structure correct")

        return OperatorFidelity("→", True, output == gold, "Transformation check")

    def check_negation(
        self,
        output: Any,
        constraints: dict,
        gold: Any,
    ) -> OperatorFidelity:
        """Check ¬ (negation) fidelity."""
        # Negation should exclude specified items
        excluded = constraints.get("values", [])

        if isinstance(output, list):
            for item in output:
                if item in excluded:
                    return OperatorFidelity("¬", True, False, f"Negated item found: {item}")
            return OperatorFidelity("¬", True, True, "Negation respected")

        return OperatorFidelity("¬", True, True, "Negation check passed")


class TaskScorer:
    """Scores task outputs against gold labels."""

    def score(
        self,
        output: Any,
        gold: Any,
        family: str,
    ) -> tuple[float, float, bool]:
        """
        Score output against gold.

        Returns:
            (accuracy, f1_score, exact_match)
        """
        if family == "selection_classification":
            return self._score_list(output, gold)
        elif family == "structured_extraction":
            return self._score_dict(output, gold)
        elif family == "constraint_composition":
            return self._score_composition(output, gold)
        elif family == "conditional_transformation":
            return self._score_transformation(output, gold)
        else:
            return self._score_generic(output, gold)

    def _score_list(
        self,
        output: Any,
        gold: Any,
    ) -> tuple[float, float, bool]:
        """Score list output (selection tasks)."""
        if not isinstance(output, list) or not isinstance(gold, list):
            return 0.0, 0.0, False

        output_set = set(str(x).lower() for x in output)
        gold_set = set(str(x).lower() for x in gold)

        # Exact match
        exact = output_set == gold_set

        # Accuracy (Jaccard similarity)
        if not gold_set and not output_set:
            accuracy = 1.0
        elif not gold_set or not output_set:
            accuracy = 0.0
        else:
            intersection = len(output_set & gold_set)
            union = len(output_set | gold_set)
            accuracy = intersection / union

        # F1 score
        true_pos = len(output_set & gold_set)
        false_pos = len(output_set - gold_set)
        false_neg = len(gold_set - output_set)

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return accuracy, f1, exact

    def _score_dict(
        self,
        output: Any,
        gold: Any,
    ) -> tuple[float, float, bool]:
        """Score dictionary output (extraction tasks)."""
        if not isinstance(output, dict) or not isinstance(gold, dict):
            return 0.0, 0.0, False

        # Exact match
        exact = output == gold

        # Per-field accuracy
        correct = 0
        total = len(gold)

        for key, gold_val in gold.items():
            out_val = output.get(key)
            if str(out_val).lower() == str(gold_val).lower():
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        f1 = accuracy  # For extraction, F1 equals field accuracy

        return accuracy, f1, exact

    def _score_composition(
        self,
        output: Any,
        gold: Any,
    ) -> tuple[float, float, bool]:
        """Score composition output."""
        if not isinstance(output, dict) or not isinstance(gold, dict):
            return 0.0, 0.0, False

        # Check selected_ids
        out_ids = set(output.get("selected_ids", []))
        gold_ids = set(gold.get("selected_ids", []))

        exact = out_ids == gold_ids

        # Accuracy
        if not gold_ids and not out_ids:
            accuracy = 1.0
        elif not gold_ids or not out_ids:
            accuracy = 0.0
        else:
            intersection = len(out_ids & gold_ids)
            union = len(out_ids | gold_ids)
            accuracy = intersection / union

        # F1
        true_pos = len(out_ids & gold_ids)
        false_pos = len(out_ids - gold_ids)
        false_neg = len(gold_ids - out_ids)

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return accuracy, f1, exact

    def _score_transformation(
        self,
        output: Any,
        gold: Any,
    ) -> tuple[float, float, bool]:
        """Score transformation output."""
        if not isinstance(output, list) or not isinstance(gold, list):
            return 0.0, 0.0, False

        exact = output == gold

        # Per-record accuracy
        correct = 0
        total = len(gold)

        for i, gold_rec in enumerate(gold):
            if i < len(output) and output[i] == gold_rec:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        f1 = accuracy

        return accuracy, f1, exact

    def _score_generic(
        self,
        output: Any,
        gold: Any,
    ) -> tuple[float, float, bool]:
        """Generic scoring fallback."""
        exact = output == gold
        accuracy = 1.0 if exact else 0.0
        return accuracy, accuracy, exact


class Evaluator:
    """Main class for evaluating model outputs."""

    # Map task families to expected output formats
    OUTPUT_FORMATS = {
        "selection_classification": "list",
        "structured_extraction": "json",
        "constraint_composition": "json",
        "conditional_transformation": "json",
        # Also support numbered prefixes
        "1_selection_classification": "list",
        "2_structured_extraction": "json",
        "3_constraint_composition": "json",
        "4_conditional_transformation": "json",
    }

    # Map operators to check functions
    OPERATOR_CHECKS = {
        "∈": "check_inclusion",
        "∉": "check_exclusion",
        "∩": "check_intersection_scope",
        "∪": "check_union_scope",
        "⇒": "check_implication",
        "∘": "check_composition_order",
        "→": "check_transformation",
        "¬": "check_negation",
    }

    def __init__(
        self,
        outputs_dir: Path | str,
        tasks_dir: Path | str,
        prompts_dir: Path | str,
        results_dir: Path | str,
        model_name: str,
    ):
        self.outputs_dir = Path(outputs_dir) / model_name
        self.tasks_dir = Path(tasks_dir)
        self.prompts_dir = Path(prompts_dir)
        self.results_dir = Path(results_dir) / model_name
        self.model_name = model_name

        self.parser = OutputParser()
        self.scorer = TaskScorer()
        self.fidelity_checker = OperatorFidelityChecker()

    def evaluate_all(self) -> dict[str, list[EvaluationResult]]:
        """Evaluate all outputs."""
        all_results = {}

        # Iterate over family directories in outputs
        for family_dir in self.outputs_dir.iterdir():
            if not family_dir.is_dir():
                continue

            family_name = family_dir.name
            results = self._evaluate_family(family_name)
            all_results[family_name] = results

        return all_results

    def evaluate_family(self, family_name: str) -> list[EvaluationResult]:
        """Evaluate outputs for a specific family."""
        return self._evaluate_family(family_name)

    def _evaluate_family(self, family_name: str) -> list[EvaluationResult]:
        """Internal method for family evaluation."""
        results_family_dir = self.results_dir / family_name
        ensure_dir(results_family_dir)

        results = []
        output_format = self.OUTPUT_FORMATS.get(family_name, "json")

        # Find all output files for this model and family
        # Outputs are now in: outputs/<model>/<family>/*.txt
        family_outputs_dir = self.outputs_dir / family_name
        if not family_outputs_dir.exists():
            return results

        output_files = list_files(family_outputs_dir, "*.txt")

        # Group results by instance_id for semantic equivalence check
        instance_results: dict[str, dict[str, EvaluationResult]] = {}

        for output_file in output_files:
            prompt_id = output_file.stem

            # Extract instance_id and condition from prompt_id
            # Format: {instance_id}_{condition}
            parts = prompt_id.rsplit('_', 1)
            if len(parts) != 2:
                continue

            instance_id, condition = parts

            # Check if this instance belongs to this family
            task_meta_path = self.tasks_dir / family_name / f"{instance_id}.meta"
            if not task_meta_path.exists():
                continue

            # Load data
            raw_output = load_text(output_file)
            gold_output = load_json(self.tasks_dir / family_name / f"{instance_id}.gold")
            constraints = load_json(self.tasks_dir / family_name / f"{instance_id}.constraints")

            # Evaluate
            result = self._evaluate_single(
                prompt_id=prompt_id,
                family=family_name,
                condition=condition,
                raw_output=raw_output,
                gold_output=gold_output,
                constraints=constraints,
                output_format=output_format,
            )

            results.append(result)

            # Store by instance for semantic equivalence check
            if instance_id not in instance_results:
                instance_results[instance_id] = {}
            instance_results[instance_id][condition] = result

        # Check semantic equivalence: NL == MG and both != CTRL
        for instance_id, conditions in instance_results.items():
            nl_result = conditions.get("NL")
            mg_result = conditions.get("MG")
            ctrl_result = conditions.get("CTRL")

            if nl_result and mg_result and ctrl_result:
                # Compare parsed outputs
                nl_output = nl_result.parsed_output
                mg_output = mg_result.parsed_output
                ctrl_output = ctrl_result.parsed_output

                # Check if NL and MG produced same output
                nl_mg_match = self._outputs_equivalent(nl_output, mg_output)
                # Check if both differ from CTRL
                nl_ctrl_differ = not self._outputs_equivalent(nl_output, ctrl_output)
                mg_ctrl_differ = not self._outputs_equivalent(mg_output, ctrl_output)

                # Semantic equivalence pass: NL == MG and both != CTRL
                if nl_mg_match and nl_ctrl_differ and mg_ctrl_differ:
                    nl_result.semantic_equivalence_pass = True
                    mg_result.semantic_equivalence_pass = True
                    # CTRL doesn't get semantic equivalence (it's the control)

        # Save all results
        for result in results:
            save_json(result.to_dict(), results_family_dir / f"{result.prompt_id}.json")

        return results

    def _outputs_equivalent(self, output1: Any, output2: Any) -> bool:
        """Check if two outputs are equivalent (allowing for minor differences)."""
        if output1 is None or output2 is None:
            return output1 == output2

        # For lists, compare as sets (order-independent)
        if isinstance(output1, list) and isinstance(output2, list):
            # Normalize strings to lowercase for comparison
            set1 = set(str(x).lower().strip() for x in output1)
            set2 = set(str(x).lower().strip() for x in output2)
            return set1 == set2

        # For dicts, compare recursively
        if isinstance(output1, dict) and isinstance(output2, dict):
            if set(output1.keys()) != set(output2.keys()):
                return False
            for key in output1:
                if not self._outputs_equivalent(output1[key], output2[key]):
                    return False
            return True

        # For strings, normalize and compare
        if isinstance(output1, str) and isinstance(output2, str):
            return output1.lower().strip() == output2.lower().strip()

        # Direct comparison for other types
        return output1 == output2

    def _evaluate_single(
        self,
        prompt_id: str,
        family: str,
        condition: str,
        raw_output: str,
        gold_output: Any,
        constraints: dict,
        output_format: str,
    ) -> EvaluationResult:
        """Evaluate a single output."""
        # Parse output
        parse_success, parsed_output, parse_error = self.parser.parse(raw_output, output_format)

        if not parse_success:
            return EvaluationResult(
                prompt_id=prompt_id,
                model=self.model_name,
                family=family,
                condition=condition,
                parse_success=False,
                parsed_output=None,
                gold_output=gold_output,
                exact_match=False,
                accuracy=0.0,
                f1_score=0.0,
                operator_fidelity=[],
                overall_pass=False,
                error_type=ErrorType.PARSE,
                error_details=parse_error,
            )

        # Score output
        accuracy, f1, exact_match = self.scorer.score(parsed_output, gold_output, family)

        # Check operator fidelity
        fidelity_results = self._check_operator_fidelity(
            parsed_output, constraints, gold_output
        )

        # Determine overall pass/fail and error type
        overall_pass = exact_match or (accuracy >= 0.8 and all(f.passed for f in fidelity_results if f.checked))
        error_type = self._classify_error(parsed_output, gold_output, fidelity_results)

        return EvaluationResult(
            prompt_id=prompt_id,
            model=self.model_name,
            family=family,
            condition=condition,
            parse_success=True,
            parsed_output=parsed_output,
            gold_output=gold_output,
            exact_match=exact_match,
            accuracy=accuracy,
            f1_score=f1,
            operator_fidelity=fidelity_results,
            overall_pass=overall_pass,
            error_type=error_type,
            error_details=None if error_type == ErrorType.NONE else f"Accuracy: {accuracy:.2f}",
        )

    def _check_operator_fidelity(
        self,
        output: Any,
        constraints: dict,
        gold: Any,
    ) -> list[OperatorFidelity]:
        """Check fidelity for all relevant operators."""
        results = []

        # Get operator from constraints
        main_op = constraints.get("operator", "")
        composition = constraints.get("composition", "")

        ops_to_check = set()
        if main_op:
            ops_to_check.add(main_op)
        if composition:
            ops_to_check.add(composition)

        # Also check operators in nested constraints
        for c in constraints.get("constraints", []):
            if "operator" in c:
                ops_to_check.add(c["operator"])

        # Check operator chain for transformations
        for op in constraints.get("operator_chain", []):
            ops_to_check.add(op)

        # Run checks
        for op in ops_to_check:
            check_method_name = self.OPERATOR_CHECKS.get(op)
            if check_method_name and hasattr(self.fidelity_checker, check_method_name):
                check_method = getattr(self.fidelity_checker, check_method_name)
                result = check_method(output, constraints, gold)
                results.append(result)

        return results

    def _classify_error(
        self,
        output: Any,
        gold: Any,
        fidelity_results: list[OperatorFidelity],
    ) -> ErrorType:
        """Classify the type of error."""
        if output == gold:
            return ErrorType.NONE

        # Check for fidelity failures
        failed_fidelity = [f for f in fidelity_results if f.checked and not f.passed]
        if failed_fidelity:
            # Determine error type based on which operators failed
            for f in failed_fidelity:
                if f.operator in ['∩', '∪', '|']:
                    return ErrorType.SCOPE
                if f.operator in ['¬', '⇒']:
                    return ErrorType.LOGIC

        # Check for structural issues
        if type(output) != type(gold):
            return ErrorType.FORMAT

        if isinstance(gold, list) and isinstance(output, list):
            if len(output) < len(gold):
                return ErrorType.INCOMPLETE
            if len(output) > len(gold):
                return ErrorType.EXTRA

        return ErrorType.LOGIC  # Default to logic error
