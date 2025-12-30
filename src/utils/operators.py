"""MetaGlyph operator definitions and registry."""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Any


class OperatorCategory(Enum):
    """Categories of MetaGlyph operators."""
    TRANSFORMATION = "transformation"
    SET = "set"
    LOGICAL = "logical"
    SCOPE = "scope"


@dataclass
class Operator:
    """Definition of a MetaGlyph operator."""
    symbol: str
    name: str
    category: OperatorCategory
    description: str
    nl_equivalent: str
    fidelity_check: str  # Name of evaluation function


OPERATORS = {
    # Transformation and rules
    "→": Operator("→", "maps_to", OperatorCategory.TRANSFORMATION,
                  "Direct transformation", "transforms to", "check_transformation"),
    "⇒": Operator("⇒", "implies", OperatorCategory.TRANSFORMATION,
                  "Conditional rule", "if...then", "check_implication"),
    "∘": Operator("∘", "compose", OperatorCategory.TRANSFORMATION,
                  "Function composition", "then apply", "check_composition_order"),
    "↦": Operator("↦", "maps_element", OperatorCategory.TRANSFORMATION,
                  "Element mapping", "each maps to", "check_element_mapping"),

    # Set and constraints
    "∈": Operator("∈", "in", OperatorCategory.SET,
                  "Set membership", "is in / belongs to", "check_inclusion"),
    "∉": Operator("∉", "not_in", OperatorCategory.SET,
                  "Set non-membership", "is not in", "check_exclusion"),
    "⊆": Operator("⊆", "subset", OperatorCategory.SET,
                  "Subset relation", "is a subset of", "check_subset"),
    "∩": Operator("∩", "intersect", OperatorCategory.SET,
                  "Set intersection", "and also / both", "check_intersection_scope"),
    "∪": Operator("∪", "union", OperatorCategory.SET,
                  "Set union", "or / either", "check_union_scope"),

    # Logical control
    "¬": Operator("¬", "not", OperatorCategory.LOGICAL,
                  "Negation", "not / exclude", "check_negation"),
    "∀": Operator("∀", "forall", OperatorCategory.LOGICAL,
                  "Universal quantifier", "for all / every", "check_universal"),
    "∃": Operator("∃", "exists", OperatorCategory.LOGICAL,
                  "Existential quantifier", "there exists / some", "check_existential"),

    # Scope restriction
    "|": Operator("|", "given", OperatorCategory.SCOPE,
                  "Conditional scope", "given / where", "check_scope_restriction"),
}


class OperatorRegistry:
    """Registry for operator lookup and validation."""

    def __init__(self):
        self._operators = OPERATORS.copy()

    def get(self, symbol: str) -> Operator | None:
        """Get operator by symbol."""
        return self._operators.get(symbol)

    def get_by_category(self, category: OperatorCategory) -> list[Operator]:
        """Get all operators in a category."""
        return [op for op in self._operators.values() if op.category == category]

    def list_symbols(self) -> list[str]:
        """List all operator symbols."""
        return list(self._operators.keys())

    def extract_operators(self, text: str) -> list[Operator]:
        """Extract all operators present in text."""
        found = []
        for symbol, op in self._operators.items():
            if symbol in text:
                found.append(op)
        return found

    def validate_instruction(self, instruction: str) -> dict[str, Any]:
        """Validate that instruction uses operators correctly."""
        operators = self.extract_operators(instruction)
        return {
            "operators_found": [op.symbol for op in operators],
            "categories": list(set(op.category.value for op in operators)),
            "count": len(operators),
        }
