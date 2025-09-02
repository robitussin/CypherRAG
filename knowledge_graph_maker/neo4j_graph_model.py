import os
from neomodel import db
from neo4j import GraphDatabase
from contextlib import contextmanager
from neomodel import install_labels
from neomodel import (
    StructuredNode,
    StringProperty,
    RelationshipTo,
    StructuredRel,
    JSONProperty,
    IntegerProperty
)
from typing import List
from .types import Edge, Node

from dotenv import dotenv_values

config = dotenv_values(".env")

@contextmanager
def neo4jDb():

    if config:       
        username = config["NEO4J_USERNAME"]
        password = config["NEO4J_PASSWORD"]
        uri = config["NEO4J_URI"]
    else:
        username = os.environ["NEO4J_USERNAME"]
        password = os.environ["NEO4J_PASSWORD"]
        uri = os.environ["NEO4J_URI"]

    driver = GraphDatabase().driver(uri, auth=(username, password))

    try:
        db.set_connection(driver=driver)
        yield db
    finally:
        # Code to release resource, e.g.:
        db.close_connection()
        

class Relationship(StructuredRel):
    description = StringProperty()
    metadata = JSONProperty()
    order = IntegerProperty()

class BaseEntity(StructuredNode):
    name = StringProperty(required=True, unique_index=True)
    label = StringProperty(required=True)  # ontology label (e.g., Person, Place)
    properties = JSONProperty(default={})
    relationship = RelationshipTo('BaseEntity', "RELATED", model=Relationship)


ENTITY_CLASSES = {}
def get_entity_class(label: str):
    """
    Returns a neomodel StructuredNode subclass for the given ontology label.
    Creates one dynamically if it does not exist yet.
    """
    if label not in ENTITY_CLASSES:
        ENTITY_CLASSES[label] = type(
            label,  # class name
            (BaseEntity,),  # parent class
            {}  # extra attributes can be injected here if needed
        )
    return ENTITY_CLASSES[label]

class Neo4jGraphModel:
    _edges: List[Edge]
    _create_indices: bool = False

    def __init__(self, edges: List[Edge], create_indices: bool = False):
        self._edges = edges
        self._create_indices = create_indices

    def migrate(self):
        if self._create_indices:
            with neo4jDb as db:
                # install indices only once
                for label in ENTITY_CLASSES.values():
                    install_labels(label, BaseEntity.relationship.definition)

    def save(self):
        count = 0
        for edge in self._edges:
            with neo4jDb() as db:
                with db.transaction:
                    # Get correct classes for source and target nodes
                    Entity1Class = get_entity_class(edge.node_1.label)
                    Entity2Class = get_entity_class(edge.node_2.label)

                    # Create or fetch nodes (unwrap list!)
                    entity_1 = Entity1Class.get_or_create({
                        "name": edge.node_1.name,
                        "label": edge.node_1.label,
                    })[0]

                    entity_2 = Entity2Class.get_or_create({
                        "name": edge.node_2.name,
                        "label": edge.node_2.label,
                    })[0]

                    # ✅ Ensure properties are merged/updated
                    if edge.node_1.properties:
                        entity_1.properties.update(edge.node_1.properties)
                        entity_1.save()

                    if edge.node_2.properties:
                        entity_2.properties.update(edge.node_2.properties)
                        entity_2.save()

                    entity_1.relationship.connect(
                        entity_2,
                        {
                            "description": edge.relationship,
                            **edge.model_dump(exclude=["description"]),
                        },
                    )
                    count += 1
        return count
