"""Validate LLM-extracted graph documents against an approved graph schema.

Schema discovery and schema enforcement are intentionally separate:
- An LLM may suggest candidate types during schema discovery.
- Application code supplies the approved node types and relationship triples here.
- Extracted graph documents must pass validation before they are persisted.

A relationship rule is a directed triple: (source_node_type, relationship_type,
target_node_type). An empty relationship-rule set means no relationships are
approved; it does not mean "allow any relationship".
"""

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RelationshipRule:
    """An approved directed relationship between two node types."""

    source: str
    relationship: str
    target: str


@dataclass(frozen=True)
class GraphSchema:
    """The approved node types and directed relationship rules."""

    node_types: frozenset[str]
    relationships: frozenset[RelationshipRule]


class GraphSchemaValidationError(ValueError):
    """Raised when extracted graph data violates the approved schema."""


def parse_schema(
    allowed_nodes: Iterable[str],
    allowed_relationships: Iterable[Sequence[str]],
) -> GraphSchema:
    """Normalize and validate an approved schema configuration.

    Args:
        allowed_nodes: Approved node labels/types, such as Order.
        allowed_relationships: Directed triples in the form
            (source_type, relationship_type, target_type).

    Raises:
        ValueError: If a type is blank, a relationship is not a triple, or a
            relationship references a node type that is not approved.
    """
    node_types = frozenset(
        node.strip()
        for node in allowed_nodes
        if isinstance(node, str) and node.strip()
    )

    relationships: set[RelationshipRule] = set()
    for index, raw_rule in enumerate(allowed_relationships):
        if isinstance(raw_rule, str) or len(raw_rule) != 3:
            raise ValueError(
                f"Relationship rule at index {index} must contain exactly "
                "(source_type, relationship_type, target_type)."
            )

        source, relationship, target = raw_rule
        if not all(
            isinstance(value, str) and value.strip()
            for value in (source, relationship, target)
        ):
            raise ValueError(
                f"Relationship rule at index {index} contains a blank or "
                "non-string value."
            )

        relationships.add(
            RelationshipRule(
                source=source.strip(),
                relationship=relationship.strip(),
                target=target.strip(),
            )
        )

    for rule in relationships:
        if rule.source not in node_types:
            raise ValueError(
                f"Relationship source type {rule.source!r} is not an "
                "approved node type."
            )
        if rule.target not in node_types:
            raise ValueError(
                f"Relationship target type {rule.target!r} is not an "
                "approved node type."
            )

    return GraphSchema(
        node_types=node_types,
        relationships=frozenset(relationships),
    )


def validate_graph_documents(
    graph_documents: Iterable[object],
    schema: GraphSchema,
) -> None:
    """Raise if any extracted node or relationship violates schema.

    Expects GraphDocument-like objects with nodes and relationships
    attributes. It does not modify or persist graph data.

    Every relationship must match an approved directed triple. Validating
    endpoint node types alone is not enough: the relationship type and
    direction must also be approved.
    """
    errors: list[str] = []

    for doc_index, graph_doc in enumerate(graph_documents):
        for node_index, node in enumerate(graph_doc.nodes):
            node_type = getattr(node, "type", None)
            if (
                not isinstance(node_type, str)
                or node_type.strip() not in schema.node_types
            ):
                errors.append(
                    f"Document {doc_index}, node {node_index}: unapproved "
                    f"node type {node_type!r}."
                )

        for rel_index, relationship in enumerate(graph_doc.relationships):
            source_type = getattr(
                getattr(relationship, "source", None), "type", None
            )
            relationship_type = getattr(relationship, "type", None)
            target_type = getattr(
                getattr(relationship, "target", None), "type", None
            )

            rule = RelationshipRule(
                source=source_type.strip()
                if isinstance(source_type, str)
                else str(source_type),
                relationship=relationship_type.strip()
                if isinstance(relationship_type, str)
                else str(relationship_type),
                target=target_type.strip()
                if isinstance(target_type, str)
                else str(target_type),
            )

            if rule.source not in schema.node_types:
                errors.append(
                    f"Document {doc_index}, relationship {rel_index}: "
                    f"unapproved source node type {source_type!r}."
                )
            if rule.target not in schema.node_types:
                errors.append(
                    f"Document {doc_index}, relationship {rel_index}: "
                    f"unapproved target node type {target_type!r}."
                )
            if rule not in schema.relationships:
                errors.append(
                    f"Document {doc_index}, relationship {rel_index}: "
                    f"unapproved relationship "
                    f"{rule.source}-[{rule.relationship}]->{rule.target}."
                )

    if errors:
        displayed_errors = "\n".join(f"- {error}" for error in errors[:50])
        remainder = len(errors) - 50
        if remainder > 0:
            displayed_errors += (
                f"\n- ... and {remainder} more validation error(s)."
            )
        raise GraphSchemaValidationError(
            "Extracted graph data failed approved-schema validation:\n"
            + displayed_errors
        )
