"""Stage 1: Dataset & Task Specification.

This module creates task instances with gold labels for each task family.
It is model-agnostic and runs once to generate all task artifacts.

Task families:
- selection_classification: Select/classify items based on criteria
- structured_extraction: Extract structured data from text
- constraint_composition: Apply composed constraints
- conditional_transformation: Transform based on conditions

Artifacts produced:
- tasks/<family>/<instance_id>.input  - Raw input text
- tasks/<family>/<instance_id>.gold   - Expected output (JSON)
- tasks/<family>/<instance_id>.constraints - Operator constraints (JSON)
"""

import uuid
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
from abc import ABC, abstractmethod

from ..utils.io_utils import save_json, save_text, ensure_dir


@dataclass
class TaskInstance:
    """A single task instance with input, gold output, and constraints."""
    instance_id: str
    family: str
    input_text: str
    gold_output: Any
    constraints: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TaskFamily:
    """Definition of a task family."""
    name: str
    operators: list[str]
    output_format: str  # "json", "list", "labels"
    description: str


# Define the four task families from the paper
TASK_FAMILIES = {
    "selection_classification": TaskFamily(
        name="selection_classification",
        operators=["∈", "∉", "¬", "∩", "∪"],
        output_format="list",
        description="Select or classify items based on set membership and logical criteria"
    ),
    "structured_extraction": TaskFamily(
        name="structured_extraction",
        operators=["∈", "→", "↦", "|"],
        output_format="json",
        description="Extract structured information with field mappings"
    ),
    "constraint_composition": TaskFamily(
        name="constraint_composition",
        operators=["∩", "∪", "¬", "⊆", "∀", "∃"],
        output_format="json",
        description="Apply multiple composed constraints to filter/transform data"
    ),
    "conditional_transformation": TaskFamily(
        name="conditional_transformation",
        operators=["⇒", "∘", "|", "→"],
        output_format="json",
        description="Apply conditional rules and chained transformations"
    ),
}


class TaskGenerator(ABC):
    """Abstract base class for task instance generators."""

    def __init__(self, family: TaskFamily, seed: int = 42):
        self.family = family
        self.rng = random.Random(seed)

    @abstractmethod
    def generate_instance(self, instance_num: int) -> TaskInstance:
        """Generate a single task instance."""
        pass

    def generate_batch(self, count: int) -> list[TaskInstance]:
        """Generate multiple task instances."""
        return [self.generate_instance(i) for i in range(count)]


class SelectionClassificationGenerator(TaskGenerator):
    """Generator for selection/classification tasks."""

    # Sample data pools for generating realistic tasks
    ENTITIES = {
        "animals": [
            ("dog", {"type": "mammal", "domestic": True, "size": "medium"}),
            ("cat", {"type": "mammal", "domestic": True, "size": "small"}),
            ("elephant", {"type": "mammal", "domestic": False, "size": "large"}),
            ("eagle", {"type": "bird", "domestic": False, "size": "medium"}),
            ("salmon", {"type": "fish", "domestic": False, "size": "medium"}),
            ("snake", {"type": "reptile", "domestic": False, "size": "medium"}),
            ("hamster", {"type": "mammal", "domestic": True, "size": "small"}),
            ("lion", {"type": "mammal", "domestic": False, "size": "large"}),
            ("penguin", {"type": "bird", "domestic": False, "size": "medium"}),
            ("goldfish", {"type": "fish", "domestic": True, "size": "small"}),
            ("turtle", {"type": "reptile", "domestic": True, "size": "small"}),
            ("whale", {"type": "mammal", "domestic": False, "size": "large"}),
            ("parrot", {"type": "bird", "domestic": True, "size": "small"}),
            ("shark", {"type": "fish", "domestic": False, "size": "large"}),
            ("crocodile", {"type": "reptile", "domestic": False, "size": "large"}),
        ],
        "products": [
            ("laptop", {"category": "electronics", "price_range": "high", "portable": True}),
            ("smartphone", {"category": "electronics", "price_range": "high", "portable": True}),
            ("desk", {"category": "furniture", "price_range": "medium", "portable": False}),
            ("chair", {"category": "furniture", "price_range": "medium", "portable": True}),
            ("notebook", {"category": "stationery", "price_range": "low", "portable": True}),
            ("pen", {"category": "stationery", "price_range": "low", "portable": True}),
            ("refrigerator", {"category": "appliances", "price_range": "high", "portable": False}),
            ("microwave", {"category": "appliances", "price_range": "medium", "portable": True}),
            ("headphones", {"category": "electronics", "price_range": "medium", "portable": True}),
            ("bookshelf", {"category": "furniture", "price_range": "medium", "portable": False}),
            ("tablet", {"category": "electronics", "price_range": "high", "portable": True}),
            ("stapler", {"category": "stationery", "price_range": "low", "portable": True}),
        ],
    }

    def __init__(self, seed: int = 42):
        super().__init__(TASK_FAMILIES["selection_classification"], seed)

    def _generate_entity_descriptions(self, domain: str, entities: list) -> str:
        """Generate verbose text describing entities."""
        lines = []
        for name, attrs in entities:
            attr_desc = ", ".join(f"{k}: {v}" for k, v in attrs.items())
            lines.append(f"- {name.capitalize()}: {attr_desc}")

        # Add padding text to reach target token range
        intro = f"The following is a comprehensive list of {domain} with their attributes. "
        intro += "Each entry contains detailed information about the item's properties. "
        intro += "Please review all entries carefully before making any selections.\n\n"

        return intro + "\n".join(lines)

    def generate_instance(self, instance_num: int) -> TaskInstance:
        """Generate a selection/classification task instance."""
        instance_id = f"sel_{instance_num:04d}"

        # Choose domain and sample entities
        domain = self.rng.choice(list(self.ENTITIES.keys()))
        all_entities = self.ENTITIES[domain]
        sampled = self.rng.sample(all_entities, min(10, len(all_entities)))

        # Generate constraint criteria
        attr_key = self.rng.choice(list(sampled[0][1].keys()))
        attr_values = list(set(e[1][attr_key] for e in sampled))
        target_value = self.rng.choice(attr_values)

        # Determine which operator pattern to use
        pattern = self.rng.choice(["inclusion", "exclusion", "intersection"])

        if pattern == "inclusion":
            # x ∈ {items where attr = value}
            gold = [name for name, attrs in sampled if attrs[attr_key] == target_value]
            constraints = {
                "operator": "∈",
                "attribute": attr_key,
                "value": target_value,
                "expected_count": len(gold),
            }
        elif pattern == "exclusion":
            # x ∉ {items where attr = value} i.e., ¬(x ∈ ...)
            gold = [name for name, attrs in sampled if attrs[attr_key] != target_value]
            constraints = {
                "operator": "∉",
                "attribute": attr_key,
                "excluded_value": target_value,
                "expected_count": len(gold),
            }
        else:  # intersection
            # Two criteria must both be satisfied
            attr_key2 = self.rng.choice([k for k in sampled[0][1].keys() if k != attr_key])
            attr_values2 = list(set(e[1][attr_key2] for e in sampled))
            target_value2 = self.rng.choice(attr_values2)
            gold = [
                name for name, attrs in sampled
                if attrs[attr_key] == target_value and attrs[attr_key2] == target_value2
            ]
            constraints = {
                "operator": "∩",
                "criteria": [
                    {"attribute": attr_key, "value": target_value},
                    {"attribute": attr_key2, "value": target_value2},
                ],
                "expected_count": len(gold),
            }

        input_text = self._generate_entity_descriptions(domain, sampled)

        # Expand input to target length (1k-5k tokens approx)
        expansion = self._generate_expansion_text(domain, len(input_text))
        input_text = input_text + "\n\n" + expansion

        return TaskInstance(
            instance_id=instance_id,
            family=self.family.name,
            input_text=input_text,
            gold_output=sorted(gold),
            constraints=constraints,
            metadata={
                "domain": domain,
                "entity_count": len(sampled),
                "pattern": pattern,
            }
        )

    def _generate_expansion_text(self, domain: str, current_len: int) -> str:
        """Generate additional context text to reach target length."""
        expansions = {
            "animals": [
                "Note: Classification should consider biological taxonomy.",
                "Domestic status indicates whether the animal is commonly kept as a pet.",
                "Size categories: small (<10kg), medium (10-100kg), large (>100kg).",
                "This dataset represents a sample of common animals across categories.",
            ],
            "products": [
                "Price ranges: low (<$50), medium ($50-$500), high (>$500).",
                "Portability indicates whether the item can be easily carried by hand.",
                "Categories are based on primary use case and retail classification.",
                "This inventory represents typical items in a department store.",
            ],
        }

        base_text = "\n".join(expansions.get(domain, expansions["products"]))

        # Repeat to reach target length
        target_chars = 4000  # Approximate for ~1k tokens
        while len(base_text) < target_chars - current_len:
            base_text += "\n" + base_text

        return base_text[:target_chars - current_len]


class StructuredExtractionGenerator(TaskGenerator):
    """Generator for structured extraction tasks."""

    DOCUMENT_TEMPLATES = [
        {
            "type": "email",
            "template": """From: {sender}
To: {recipient}
Subject: {subject}
Date: {date}

Dear {recipient_name},

{body}

Best regards,
{sender_name}
{sender_title}
{company}""",
            "fields": ["sender", "recipient", "subject", "date", "sender_name", "sender_title", "company"],
        },
        {
            "type": "invoice",
            "template": """INVOICE #{invoice_num}
Date: {date}
Due Date: {due_date}

Bill To:
{customer_name}
{customer_address}

Items:
{items}

Subtotal: ${subtotal}
Tax ({tax_rate}%): ${tax}
Total: ${total}

Payment Terms: {payment_terms}""",
            "fields": ["invoice_num", "date", "due_date", "customer_name", "total", "payment_terms"],
        },
    ]

    def __init__(self, seed: int = 42):
        super().__init__(TASK_FAMILIES["structured_extraction"], seed)

    def generate_instance(self, instance_num: int) -> TaskInstance:
        """Generate a structured extraction task instance."""
        instance_id = f"ext_{instance_num:04d}"

        # Choose document type
        doc_type = self.rng.choice(self.DOCUMENT_TEMPLATES)

        # Generate values for template
        values = self._generate_values(doc_type["type"])

        # Fill template
        try:
            input_text = doc_type["template"].format(**values)
        except KeyError:
            input_text = doc_type["template"]
            for k, v in values.items():
                input_text = input_text.replace(f"{{{k}}}", str(v))

        # Add surrounding context
        input_text = self._add_context(input_text, doc_type["type"])

        # Select fields to extract
        extract_fields = self.rng.sample(doc_type["fields"], min(4, len(doc_type["fields"])))

        gold_output = {field: values.get(field, "") for field in extract_fields}

        constraints = {
            "operator": "→",
            "document_type": doc_type["type"],
            "required_fields": extract_fields,
            "field_count": len(extract_fields),
        }

        return TaskInstance(
            instance_id=instance_id,
            family=self.family.name,
            input_text=input_text,
            gold_output=gold_output,
            constraints=constraints,
            metadata={
                "document_type": doc_type["type"],
                "field_count": len(extract_fields),
            }
        )

    def _generate_values(self, doc_type: str) -> dict:
        """Generate realistic values for document fields."""
        names = ["Alice Smith", "Bob Johnson", "Carol Williams", "David Brown"]
        companies = ["TechCorp Inc.", "Global Solutions", "Innovation Labs", "Digital Systems"]

        if doc_type == "email":
            sender_name = self.rng.choice(names)
            return {
                "sender": f"{sender_name.lower().replace(' ', '.')}@{self.rng.choice(companies).lower().replace(' ', '').replace('.', '')}.com",
                "recipient": f"{self.rng.choice(names).lower().replace(' ', '.')}@example.com",
                "subject": self.rng.choice(["Q4 Report", "Meeting Request", "Project Update", "Action Required"]),
                "date": f"2024-{self.rng.randint(1,12):02d}-{self.rng.randint(1,28):02d}",
                "recipient_name": self.rng.choice(names).split()[0],
                "body": "I wanted to follow up on our recent discussion regarding the project timeline. Please review the attached documents and let me know your thoughts.",
                "sender_name": sender_name,
                "sender_title": self.rng.choice(["Manager", "Director", "VP", "Analyst"]),
                "company": self.rng.choice(companies),
            }
        else:  # invoice
            subtotal = self.rng.randint(100, 10000)
            tax_rate = self.rng.choice([5, 7, 10, 15])
            tax = round(subtotal * tax_rate / 100, 2)
            return {
                "invoice_num": f"{self.rng.randint(1000, 9999)}",
                "date": f"2024-{self.rng.randint(1,12):02d}-{self.rng.randint(1,28):02d}",
                "due_date": f"2024-{self.rng.randint(1,12):02d}-{self.rng.randint(1,28):02d}",
                "customer_name": self.rng.choice(names),
                "customer_address": f"{self.rng.randint(100, 999)} Main St, City, ST {self.rng.randint(10000, 99999)}",
                "items": "1x Product A - $500\n2x Service B - $250 each",
                "subtotal": subtotal,
                "tax_rate": tax_rate,
                "tax": tax,
                "total": subtotal + tax,
                "payment_terms": self.rng.choice(["Net 30", "Net 60", "Due on Receipt"]),
            }

    def _add_context(self, text: str, doc_type: str) -> str:
        """Add surrounding context to reach target length."""
        prefix = f"The following is a {doc_type} document. Please extract the requested fields.\n\n"
        prefix += "=" * 50 + "\n"
        suffix = "\n" + "=" * 50
        suffix += "\n\nEnd of document. Extract only the fields specified in the instruction."

        # Pad to target length
        padding = "\n\n[Additional context for processing...]\n" * 50

        return prefix + text + suffix + padding


class ConstraintCompositionGenerator(TaskGenerator):
    """Generator for constraint composition tasks."""

    def __init__(self, seed: int = 42):
        super().__init__(TASK_FAMILIES["constraint_composition"], seed)

    def generate_instance(self, instance_num: int) -> TaskInstance:
        """Generate a constraint composition task instance."""
        instance_id = f"cmp_{instance_num:04d}"

        # Generate a dataset of items with multiple attributes
        items = self._generate_items(20)

        # Create composed constraints
        constraints_spec = self._generate_constraints(items)

        # Apply constraints to get gold output
        gold_output = self._apply_constraints(items, constraints_spec)

        # Format input text
        input_text = self._format_items(items)

        return TaskInstance(
            instance_id=instance_id,
            family=self.family.name,
            input_text=input_text,
            gold_output=gold_output,
            constraints=constraints_spec,
            metadata={
                "item_count": len(items),
                "constraint_count": len(constraints_spec.get("constraints", [])),
            }
        )

    def _generate_items(self, count: int) -> list[dict]:
        """Generate items with attributes."""
        categories = ["A", "B", "C"]
        statuses = ["active", "inactive", "pending"]
        priorities = ["low", "medium", "high"]

        items = []
        for i in range(count):
            items.append({
                "id": f"item_{i:03d}",
                "category": self.rng.choice(categories),
                "status": self.rng.choice(statuses),
                "priority": self.rng.choice(priorities),
                "value": self.rng.randint(1, 100),
            })
        return items

    def _generate_constraints(self, items: list[dict]) -> dict:
        """Generate composed constraints."""
        constraint_type = self.rng.choice(["intersection", "union_exclusion", "subset_check"])

        if constraint_type == "intersection":
            return {
                "composition": "∩",
                "constraints": [
                    {"operator": "∈", "field": "category", "values": ["A", "B"]},
                    {"operator": "∈", "field": "status", "values": ["active"]},
                ],
            }
        elif constraint_type == "union_exclusion":
            return {
                "composition": "∪",
                "constraints": [
                    {"operator": "∈", "field": "priority", "values": ["high"]},
                    {"operator": "¬", "field": "status", "values": ["inactive"]},
                ],
            }
        else:
            return {
                "composition": "∩",
                "constraints": [
                    {"operator": "∀", "field": "value", "condition": ">=", "threshold": 50},
                    {"operator": "∈", "field": "category", "values": ["A"]},
                ],
            }

    def _apply_constraints(self, items: list[dict], constraints_spec: dict) -> dict:
        """Apply constraints to filter items."""
        results = []

        for item in items:
            passes = self._check_item(item, constraints_spec)
            if passes:
                results.append(item["id"])

        return {"selected_ids": sorted(results), "count": len(results)}

    def _check_item(self, item: dict, spec: dict) -> bool:
        """Check if item passes all constraints."""
        composition = spec.get("composition", "∩")
        constraints = spec.get("constraints", [])

        results = []
        for c in constraints:
            op = c.get("operator")
            field = c.get("field")

            if op == "∈":
                results.append(item.get(field) in c.get("values", []))
            elif op == "¬":
                results.append(item.get(field) not in c.get("values", []))
            elif op == "∀":
                threshold = c.get("threshold", 0)
                condition = c.get("condition", ">=")
                val = item.get(field, 0)
                if condition == ">=":
                    results.append(val >= threshold)
                elif condition == "<=":
                    results.append(val <= threshold)

        if composition == "∩":
            return all(results) if results else True
        else:  # union
            return any(results) if results else False

    def _format_items(self, items: list[dict]) -> str:
        """Format items as input text."""
        lines = ["Dataset of items with attributes:", ""]
        for item in items:
            lines.append(f"ID: {item['id']}")
            lines.append(f"  Category: {item['category']}")
            lines.append(f"  Status: {item['status']}")
            lines.append(f"  Priority: {item['priority']}")
            lines.append(f"  Value: {item['value']}")
            lines.append("")

        # Pad to target length
        return "\n".join(lines) + "\n" * 100


class ConditionalTransformationGenerator(TaskGenerator):
    """Generator for conditional transformation tasks."""

    def __init__(self, seed: int = 42):
        super().__init__(TASK_FAMILIES["conditional_transformation"], seed)

    def generate_instance(self, instance_num: int) -> TaskInstance:
        """Generate a conditional transformation task instance."""
        instance_id = f"trn_{instance_num:04d}"

        # Generate source data
        records = self._generate_records(15)

        # Define transformation rules
        rules = self._generate_rules()

        # Apply transformations to get gold output
        gold_output = self._apply_transformations(records, rules)

        # Format input
        input_text = self._format_records(records)

        constraints = {
            "operator_chain": ["⇒", "∘"],
            "rules": rules,
            "rule_count": len(rules),
        }

        return TaskInstance(
            instance_id=instance_id,
            family=self.family.name,
            input_text=input_text,
            gold_output=gold_output,
            constraints=constraints,
            metadata={
                "record_count": len(records),
                "rule_count": len(rules),
            }
        )

    def _generate_records(self, count: int) -> list[dict]:
        """Generate records for transformation."""
        types = ["standard", "premium", "trial"]
        regions = ["US", "EU", "APAC"]

        records = []
        for i in range(count):
            records.append({
                "id": f"rec_{i:03d}",
                "type": self.rng.choice(types),
                "region": self.rng.choice(regions),
                "amount": self.rng.randint(10, 1000),
                "active": self.rng.choice([True, False]),
            })
        return records

    def _generate_rules(self) -> list[dict]:
        """Generate transformation rules."""
        return [
            {
                "condition": {"field": "type", "operator": "==", "value": "premium"},
                "action": {"field": "amount", "operation": "multiply", "factor": 1.1},
            },
            {
                "condition": {"field": "region", "operator": "==", "value": "EU"},
                "action": {"field": "amount", "operation": "multiply", "factor": 1.2},
            },
            {
                "condition": {"field": "active", "operator": "==", "value": False},
                "action": {"field": "status", "operation": "set", "value": "archived"},
            },
        ]

    def _apply_transformations(self, records: list[dict], rules: list[dict]) -> list[dict]:
        """Apply transformation rules to records."""
        results = []

        for record in records:
            transformed = record.copy()

            for rule in rules:
                cond = rule["condition"]
                if self._check_condition(transformed, cond):
                    self._apply_action(transformed, rule["action"])

            results.append(transformed)

        return results

    def _check_condition(self, record: dict, condition: dict) -> bool:
        """Check if record matches condition."""
        field = condition["field"]
        op = condition["operator"]
        value = condition["value"]

        record_value = record.get(field)

        if op == "==":
            return record_value == value
        elif op == "!=":
            return record_value != value
        elif op == ">":
            return record_value > value
        elif op == "<":
            return record_value < value
        return False

    def _apply_action(self, record: dict, action: dict) -> None:
        """Apply transformation action to record."""
        field = action["field"]
        operation = action["operation"]

        if operation == "multiply":
            factor = action.get("factor", 1)
            record[field] = round(record.get(field, 0) * factor, 2)
        elif operation == "set":
            record[field] = action.get("value")
        elif operation == "add":
            record[field] = record.get(field, 0) + action.get("value", 0)

    def _format_records(self, records: list[dict]) -> str:
        """Format records as input text."""
        lines = ["Records for transformation:", ""]
        for rec in records:
            lines.append(f"Record {rec['id']}:")
            for k, v in rec.items():
                if k != "id":
                    lines.append(f"  {k}: {v}")
            lines.append("")

        return "\n".join(lines) + "\n" * 80


class DatasetGenerator:
    """Main class for generating all task datasets."""

    GENERATORS = {
        "selection_classification": SelectionClassificationGenerator,
        "structured_extraction": StructuredExtractionGenerator,
        "constraint_composition": ConstraintCompositionGenerator,
        "conditional_transformation": ConditionalTransformationGenerator,
    }

    def __init__(self, output_dir: Path | str, seed: int = 42):
        self.output_dir = Path(output_dir)
        self.seed = seed

    def generate_all(self, instances_per_family: int = 50) -> dict[str, list[TaskInstance]]:
        """Generate task instances for all families."""
        all_instances = {}

        for family_name, generator_class in self.GENERATORS.items():
            generator = generator_class(seed=self.seed)
            instances = generator.generate_batch(instances_per_family)
            all_instances[family_name] = instances

            # Save artifacts
            self._save_family(family_name, instances)

        return all_instances

    def generate_family(self, family_name: str, count: int = 50) -> list[TaskInstance]:
        """Generate task instances for a specific family."""
        if family_name not in self.GENERATORS:
            raise ValueError(f"Unknown task family: {family_name}")

        generator = self.GENERATORS[family_name](seed=self.seed)
        instances = generator.generate_batch(count)
        self._save_family(family_name, instances)
        return instances

    def _save_family(self, family_name: str, instances: list[TaskInstance]) -> None:
        """Save all artifacts for a task family."""
        family_dir = self.output_dir / family_name
        ensure_dir(family_dir)

        for instance in instances:
            base_path = family_dir / instance.instance_id

            # Save input text
            save_text(instance.input_text, f"{base_path}.input")

            # Save gold output
            save_json(instance.gold_output, f"{base_path}.gold")

            # Save constraints
            save_json(instance.constraints, f"{base_path}.constraints")

            # Save full instance metadata
            save_json(instance.to_dict(), f"{base_path}.meta")

    def list_families(self) -> list[str]:
        """List available task families."""
        return list(self.GENERATORS.keys())
