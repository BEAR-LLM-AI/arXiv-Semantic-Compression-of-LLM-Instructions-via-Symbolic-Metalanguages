"""Stage 2: Prompt Construction.

This module generates three prompt variants per task instance:
- NL: Natural language instruction (verbose)
- MG: MetaGlyph symbolic instruction (compact)
- CTRL: Symbol-shaped control (same structure, broken semantics)

All variants share identical input text and output format constraints.

Artifacts produced:
- prompts/<family>/<instance_id>_NL.txt
- prompts/<family>/<instance_id>_MG.txt
- prompts/<family>/<instance_id>_CTRL.txt
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json

from ..utils.io_utils import load_json, load_text, save_json, save_text, ensure_dir, list_files
from ..utils.operators import OperatorRegistry


@dataclass
class Prompt:
    """A constructed prompt with metadata."""
    prompt_id: str
    instance_id: str
    family: str
    condition: str  # NL, MG, or CTRL
    instruction: str
    input_text: str
    output_format: str
    full_prompt: str

    def to_dict(self) -> dict:
        return asdict(self)


class InstructionGenerator:
    """Generates instruction variants for different conditions."""

    def __init__(self):
        self.operators = OperatorRegistry()

    def generate_nl(self, family: str, constraints: dict, metadata: dict) -> str:
        """Generate verbose natural language instruction."""
        generators = {
            "selection_classification": self._nl_selection,
            "structured_extraction": self._nl_extraction,
            "constraint_composition": self._nl_composition,
            "conditional_transformation": self._nl_transformation,
        }
        return generators.get(family, self._nl_generic)(constraints, metadata)

    def generate_mg(self, family: str, constraints: dict, metadata: dict) -> str:
        """Generate compact MetaGlyph symbolic instruction."""
        generators = {
            "selection_classification": self._mg_selection,
            "structured_extraction": self._mg_extraction,
            "constraint_composition": self._mg_composition,
            "conditional_transformation": self._mg_transformation,
        }
        return generators.get(family, self._mg_generic)(constraints, metadata)

    def generate_ctrl(self, family: str, constraints: dict, metadata: dict) -> str:
        """Generate symbol-shaped control with broken semantics."""
        generators = {
            "selection_classification": self._ctrl_selection,
            "structured_extraction": self._ctrl_extraction,
            "constraint_composition": self._ctrl_composition,
            "conditional_transformation": self._ctrl_transformation,
        }
        return generators.get(family, self._ctrl_generic)(constraints, metadata)

    # Selection/Classification instructions
    def _nl_selection(self, constraints: dict, metadata: dict) -> str:
        """Natural language instruction for selection tasks."""
        op = constraints.get("operator", "∈")
        domain = metadata.get("domain", "items")

        if op == "∈":
            attr = constraints.get("attribute", "type")
            value = constraints.get("value", "")
            return (
                f"Please carefully review the entire list of {domain} provided below. "
                f"Your task is to identify and select all items from this list where the "
                f"'{attr}' attribute has the value '{value}'. "
                f"Make sure to examine each entry thoroughly and include only those items "
                f"that exactly match this criterion. Return the names of all matching items "
                f"as a JSON array of strings."
            )
        elif op == "∉":
            attr = constraints.get("attribute", "type")
            excluded = constraints.get("excluded_value", "")
            return (
                f"Please carefully review the entire list of {domain} provided below. "
                f"Your task is to identify and select all items from this list where the "
                f"'{attr}' attribute does NOT have the value '{excluded}'. "
                f"In other words, exclude any items that match '{excluded}' for this attribute. "
                f"Return the names of all remaining items as a JSON array of strings."
            )
        elif op == "∩":
            criteria = constraints.get("criteria", [])
            if len(criteria) >= 2:
                c1, c2 = criteria[0], criteria[1]
                return (
                    f"Please carefully review the entire list of {domain} provided below. "
                    f"Your task is to identify items that satisfy BOTH of the following conditions: "
                    f"(1) the '{c1['attribute']}' attribute has the value '{c1['value']}', AND "
                    f"(2) the '{c2['attribute']}' attribute has the value '{c2['value']}'. "
                    f"Only include items that meet both criteria simultaneously. "
                    f"Return the names of all matching items as a JSON array of strings."
                )
        return self._nl_generic(constraints, metadata)

    def _mg_selection(self, constraints: dict, metadata: dict) -> str:
        """MetaGlyph instruction for selection tasks."""
        op = constraints.get("operator", "∈")

        if op == "∈":
            attr = constraints.get("attribute", "type")
            value = constraints.get("value", "")
            return f"SELECT x | x.{attr} ∈ {{{value}}} → [x.name]"
        elif op == "∉":
            attr = constraints.get("attribute", "type")
            excluded = constraints.get("excluded_value", "")
            return f"SELECT x | x.{attr} ∉ {{{excluded}}} → [x.name]"
        elif op == "∩":
            criteria = constraints.get("criteria", [])
            if len(criteria) >= 2:
                c1, c2 = criteria[0], criteria[1]
                return (
                    f"SELECT x | (x.{c1['attribute']} ∈ {{{c1['value']}}}) "
                    f"∩ (x.{c2['attribute']} ∈ {{{c2['value']}}}) → [x.name]"
                )
        return self._mg_generic(constraints, metadata)

    def _ctrl_selection(self, constraints: dict, metadata: dict) -> str:
        """Control instruction for selection tasks (broken semantics)."""
        op = constraints.get("operator", "∈")

        # Use symbols but with semantically incorrect structure
        if op == "∈":
            attr = constraints.get("attribute", "type")
            value = constraints.get("value", "")
            # Invert the operator semantics
            return f"SELECT x | x.{attr} ∉ {{{value}}} → [x.name]"
        elif op == "∉":
            attr = constraints.get("attribute", "type")
            excluded = constraints.get("excluded_value", "")
            return f"SELECT x | x.{attr} ∈ {{{excluded}}} → [x.name]"
        elif op == "∩":
            criteria = constraints.get("criteria", [])
            if len(criteria) >= 2:
                c1, c2 = criteria[0], criteria[1]
                # Use union instead of intersection (different semantic)
                return (
                    f"SELECT x | (x.{c1['attribute']} ∈ {{{c1['value']}}}) "
                    f"∪ (x.{c2['attribute']} ∈ {{{c2['value']}}}) → [x.name]"
                )
        return self._ctrl_generic(constraints, metadata)

    # Structured Extraction instructions
    def _nl_extraction(self, constraints: dict, metadata: dict) -> str:
        """Natural language instruction for extraction tasks."""
        doc_type = constraints.get("document_type", "document")
        fields = constraints.get("required_fields", [])
        fields_str = ", ".join(f"'{f}'" for f in fields)

        return (
            f"Please examine the {doc_type} document provided below and extract "
            f"the following specific fields: {fields_str}. "
            f"For each field, locate the corresponding value in the document and "
            f"include it in your response. If a field is not present or cannot be "
            f"determined, use null. Return the extracted data as a JSON object "
            f"with the field names as keys and the extracted values as values."
        )

    def _mg_extraction(self, constraints: dict, metadata: dict) -> str:
        """MetaGlyph instruction for extraction tasks."""
        fields = constraints.get("required_fields", [])
        mappings = ", ".join(f'"{f}" ↦ value' for f in fields)
        return f"EXTRACT doc → {{{mappings}}}"

    def _ctrl_extraction(self, constraints: dict, metadata: dict) -> str:
        """Control instruction for extraction tasks."""
        fields = constraints.get("required_fields", [])
        # Shuffle field names to break semantics while keeping structure
        import random
        shuffled = fields.copy()
        random.Random(42).shuffle(shuffled)
        mappings = ", ".join(f'"{f}" ↦ value' for f in shuffled)
        return f"EXTRACT doc → {{{mappings}}}"

    # Constraint Composition instructions
    def _nl_composition(self, constraints: dict, metadata: dict) -> str:
        """Natural language instruction for composition tasks."""
        composition = constraints.get("composition", "∩")
        constraint_list = constraints.get("constraints", [])

        conditions = []
        for c in constraint_list:
            op = c.get("operator")
            field = c.get("field")
            if op == "∈":
                values = c.get("values", [])
                conditions.append(f"the '{field}' is one of: {', '.join(str(v) for v in values)}")
            elif op == "¬":
                values = c.get("values", [])
                conditions.append(f"the '{field}' is NOT one of: {', '.join(str(v) for v in values)}")
            elif op == "∀":
                threshold = c.get("threshold", 0)
                cond = c.get("condition", ">=")
                conditions.append(f"the '{field}' is {cond} {threshold}")

        joiner = " AND " if composition == "∩" else " OR "
        conditions_str = joiner.join(conditions)

        return (
            f"Please review all items in the dataset below. Select and return the IDs "
            f"of all items where {conditions_str}. "
            f"Return the result as a JSON object with 'selected_ids' (array of IDs) "
            f"and 'count' (number of selected items)."
        )

    def _mg_composition(self, constraints: dict, metadata: dict) -> str:
        """MetaGlyph instruction for composition tasks."""
        composition = constraints.get("composition", "∩")
        constraint_list = constraints.get("constraints", [])

        parts = []
        for c in constraint_list:
            op = c.get("operator")
            field = c.get("field")
            if op == "∈":
                values = c.get("values", [])
                parts.append(f"x.{field} ∈ {{{', '.join(str(v) for v in values)}}}")
            elif op == "¬":
                values = c.get("values", [])
                parts.append(f"x.{field} ∉ {{{', '.join(str(v) for v in values)}}}")
            elif op == "∀":
                threshold = c.get("threshold", 0)
                cond = c.get("condition", ">=")
                parts.append(f"x.{field} {cond} {threshold}")

        expr = f" {composition} ".join(parts)
        return f"SELECT x.id | {expr} → {{selected_ids: [...], count: n}}"

    def _ctrl_composition(self, constraints: dict, metadata: dict) -> str:
        """Control instruction for composition tasks."""
        composition = constraints.get("composition", "∩")
        constraint_list = constraints.get("constraints", [])

        # Flip the composition operator
        flipped = "∪" if composition == "∩" else "∩"

        parts = []
        for c in constraint_list:
            op = c.get("operator")
            field = c.get("field")
            if op == "∈":
                values = c.get("values", [])
                parts.append(f"x.{field} ∈ {{{', '.join(str(v) for v in values)}}}")
            elif op == "¬":
                values = c.get("values", [])
                parts.append(f"x.{field} ∉ {{{', '.join(str(v) for v in values)}}}")
            elif op == "∀":
                threshold = c.get("threshold", 0)
                cond = c.get("condition", ">=")
                parts.append(f"x.{field} {cond} {threshold}")

        expr = f" {flipped} ".join(parts)  # Wrong composition
        return f"SELECT x.id | {expr} → {{selected_ids: [...], count: n}}"

    # Conditional Transformation instructions
    def _nl_transformation(self, constraints: dict, metadata: dict) -> str:
        """Natural language instruction for transformation tasks."""
        rules = constraints.get("rules", [])

        rule_descriptions = []
        for i, rule in enumerate(rules, 1):
            cond = rule.get("condition", {})
            action = rule.get("action", {})

            cond_str = f"{cond.get('field')} {cond.get('operator')} {cond.get('value')}"
            if action.get("operation") == "multiply":
                action_str = f"multiply {action.get('field')} by {action.get('factor')}"
            elif action.get("operation") == "set":
                action_str = f"set {action.get('field')} to '{action.get('value')}'"
            else:
                action_str = f"modify {action.get('field')}"

            rule_descriptions.append(f"Rule {i}: If {cond_str}, then {action_str}")

        rules_text = "; ".join(rule_descriptions)

        return (
            f"Please process each record in the dataset below by applying the following "
            f"transformation rules in order: {rules_text}. "
            f"For each record, check if the condition is met, and if so, apply the "
            f"corresponding transformation. Multiple rules may apply to the same record. "
            f"Return all records (transformed where applicable) as a JSON array."
        )

    def _mg_transformation(self, constraints: dict, metadata: dict) -> str:
        """MetaGlyph instruction for transformation tasks."""
        rules = constraints.get("rules", [])

        rule_parts = []
        for rule in rules:
            cond = rule.get("condition", {})
            action = rule.get("action", {})

            cond_str = f"x.{cond.get('field')} == {json.dumps(cond.get('value'))}"

            if action.get("operation") == "multiply":
                action_str = f"x.{action.get('field')} → x.{action.get('field')} × {action.get('factor')}"
            elif action.get("operation") == "set":
                action_str = f"x.{action.get('field')} → {json.dumps(action.get('value'))}"
            else:
                action_str = f"x.{action.get('field')} → modified"

            rule_parts.append(f"({cond_str} ⇒ {action_str})")

        chain = " ∘ ".join(rule_parts)
        return f"∀x: TRANSFORM x | {chain} → [x]"

    def _ctrl_transformation(self, constraints: dict, metadata: dict) -> str:
        """Control instruction for transformation tasks."""
        rules = constraints.get("rules", [])

        rule_parts = []
        for rule in rules:
            cond = rule.get("condition", {})
            action = rule.get("action", {})

            # Negate the condition to break semantics
            cond_str = f"x.{cond.get('field')} != {json.dumps(cond.get('value'))}"

            if action.get("operation") == "multiply":
                action_str = f"x.{action.get('field')} → x.{action.get('field')} × {action.get('factor')}"
            elif action.get("operation") == "set":
                action_str = f"x.{action.get('field')} → {json.dumps(action.get('value'))}"
            else:
                action_str = f"x.{action.get('field')} → modified"

            rule_parts.append(f"({cond_str} ⇒ {action_str})")

        chain = " ∘ ".join(rule_parts)
        return f"∀x: TRANSFORM x | {chain} → [x]"

    # Generic fallbacks
    def _nl_generic(self, constraints: dict, metadata: dict) -> str:
        """Generic natural language instruction."""
        return "Please process the input according to the specified constraints and return the result as JSON."

    def _mg_generic(self, constraints: dict, metadata: dict) -> str:
        """Generic MetaGlyph instruction."""
        return "PROCESS input → output"

    def _ctrl_generic(self, constraints: dict, metadata: dict) -> str:
        """Generic control instruction."""
        return "PROCESS input → output"


class PromptConstructor:
    """Main class for constructing prompts from task instances."""

    OUTPUT_FORMATS = {
        "selection_classification": "Return your answer as a JSON array of strings, e.g., [\"item1\", \"item2\"].",
        "structured_extraction": "Return your answer as a JSON object with the specified field names as keys.",
        "constraint_composition": "Return your answer as a JSON object with 'selected_ids' (array) and 'count' (integer).",
        "conditional_transformation": "Return your answer as a JSON array of transformed record objects.",
    }

    def __init__(self, tasks_dir: Path | str, output_dir: Path | str):
        self.tasks_dir = Path(tasks_dir)
        self.output_dir = Path(output_dir)
        self.instruction_gen = InstructionGenerator()

    def construct_all(self) -> dict[str, list[Prompt]]:
        """Construct prompts for all task instances."""
        all_prompts = {}

        for family_dir in self.tasks_dir.iterdir():
            if not family_dir.is_dir():
                continue

            family_name = family_dir.name
            prompts = self._construct_family(family_name)
            all_prompts[family_name] = prompts

        return all_prompts

    def construct_family(self, family_name: str) -> list[Prompt]:
        """Construct prompts for a specific task family."""
        return self._construct_family(family_name)

    def _construct_family(self, family_name: str) -> list[Prompt]:
        """Internal method to construct prompts for a family."""
        family_dir = self.tasks_dir / family_name
        output_family_dir = self.output_dir / family_name
        ensure_dir(output_family_dir)

        prompts = []

        # Find all task instances (by .meta files)
        meta_files = list_files(family_dir, "*.meta")

        for meta_file in meta_files:
            instance_id = meta_file.stem

            # Load task data
            input_text = load_text(family_dir / f"{instance_id}.input")
            constraints = load_json(family_dir / f"{instance_id}.constraints")
            metadata = load_json(meta_file).get("metadata", {})

            # Generate three prompt variants
            for condition in ["NL", "MG", "CTRL"]:
                prompt = self._construct_prompt(
                    instance_id=instance_id,
                    family=family_name,
                    condition=condition,
                    input_text=input_text,
                    constraints=constraints,
                    metadata=metadata,
                )
                prompts.append(prompt)

                # Save prompt
                self._save_prompt(prompt, output_family_dir)

        return prompts

    def _construct_prompt(
        self,
        instance_id: str,
        family: str,
        condition: str,
        input_text: str,
        constraints: dict,
        metadata: dict,
    ) -> Prompt:
        """Construct a single prompt."""
        prompt_id = f"{instance_id}_{condition}"

        # Generate instruction based on condition
        if condition == "NL":
            instruction = self.instruction_gen.generate_nl(family, constraints, metadata)
        elif condition == "MG":
            instruction = self.instruction_gen.generate_mg(family, constraints, metadata)
        else:  # CTRL
            instruction = self.instruction_gen.generate_ctrl(family, constraints, metadata)

        # Get output format constraint
        output_format = self.OUTPUT_FORMATS.get(family, "Return your answer as JSON.")

        # Assemble full prompt
        full_prompt = self._assemble_prompt(instruction, input_text, output_format)

        return Prompt(
            prompt_id=prompt_id,
            instance_id=instance_id,
            family=family,
            condition=condition,
            instruction=instruction,
            input_text=input_text,
            output_format=output_format,
            full_prompt=full_prompt,
        )

    def _assemble_prompt(self, instruction: str, input_text: str, output_format: str) -> str:
        """Assemble the full prompt from components."""
        return f"""### Instruction
{instruction}

### Input
{input_text}

### Output Format
{output_format}

### Response"""

    def _save_prompt(self, prompt: Prompt, output_dir: Path) -> None:
        """Save prompt artifacts."""
        base_path = output_dir / prompt.prompt_id

        # Save full prompt text
        save_text(prompt.full_prompt, f"{base_path}.txt")

        # Save prompt metadata
        save_json(prompt.to_dict(), f"{base_path}.json")
