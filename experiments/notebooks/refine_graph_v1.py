from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from query_graph import QueryGraph

def refine(qg: QueryGraph, entityType: str):
    
    if refactor_graph(qg):
        if create_embeddings(entityType):
            type, projection = create_projection(entityType, qg)
            listofnodes = calculateSimilarityScore(projection, qg)
            if mergeNodes(type, listofnodes, qg):
                return True
            else:
                return False   
        return False
    return False
def create_embeddings(entityType: str) -> bool:
    
    nodelabel = ""
    indexname = ""
    
    if entityType == "person":
        nodelabel = "Person"
        indexname = "person_index"
    elif entityType == "object":
        nodelabel = "Object"
        indexname = "object_index"
    elif entityType == "event":
        nodelabel = "Event"
        indexname = "event_index"
    elif entityType == "place":
        nodelabel = "Place"
        indexname = "place_index"
    elif entityType == "miscellaneous":
        nodelabel = "Miscellaneous"
        indexname = "miscellaneous_index"
    
    try:
        Neo4jVector.from_existing_graph(
            embedding=OpenAIEmbeddings(model='text-embedding-3-small'),
            url="bolt://localhost:7687",
            username="neo4j",
            password="12345678",
            index_name=indexname,
            node_label=nodelabel,
            text_node_properties=["name"],
            embedding_node_property="embedding",
        )
        return True
    except():
        return False  

def refactor_graph(qg: QueryGraph) -> bool:

    cq = """
    MATCH (en:Entity)
    WHERE en.label = 'Person'
    WITH collect(en) AS persons
    CALL apoc.refactor.rename.label("Entity", "Person", persons)
    YIELD batches, total, timeTaken, committedOperations
    RETURN batches, total, timeTaken, committedOperations;
    """
    qg._graph.query(cq)

    cq = """
    MATCH (en:Entity)
    WHERE en.label = 'Place'
    WITH collect(en) AS places
    CALL apoc.refactor.rename.label("Entity", "Place", places)
    YIELD batches, total, timeTaken, committedOperations
    RETURN batches, total, timeTaken, committedOperations;
    """
    qg._graph.query(cq)

    cq = """
    MATCH (en:Entity)
    WHERE en.label = 'Object'
    WITH collect(en) AS objects
    CALL apoc.refactor.rename.label("Entity", "Object", objects)
    YIELD batches, total, timeTaken, committedOperations
    RETURN batches, total, timeTaken, committedOperations;
    """
    qg._graph.query(cq)

    cq = """
    MATCH (en:Entity)
    WHERE en.label = 'Event'
    WITH collect(en) AS events
    CALL apoc.refactor.rename.label("Entity", "Event", events)
    YIELD batches, total, timeTaken, committedOperations
    RETURN batches, total, timeTaken, committedOperations;
    """
    qg._graph.query(cq)

    cq = """
    MATCH (en:Entity)
    WHERE en.label = 'Miscellaneous'
    WITH collect(en) AS miscellaneous
    CALL apoc.refactor.rename.label("Entity", "Miscellaneous", miscellaneous)
    YIELD batches, total, timeTaken, committedOperations
    RETURN batches, total, timeTaken, committedOperations;
    """
    qg._graph.query(cq)
    
    return True

def create_projection(entityType: str, qg: QueryGraph):
    
    type = ""
    
    if entityType == "person":
        type = "Person"
        projectionname = "personproj"
    if entityType == "event":
        type = "Event"
        projectionname = "eventproj"
    if entityType == "object":
        type = "Object"
        projectionname = "objectproj"
    if entityType == "place":
        type = "Place"
        projectionname = "placeproj"
    if entityType == "miscellaneous":
        type = "Miscellaneous"
        projectionname = "miscellaneousproj"
            
    cq = f"""
    MATCH (p:{type})
    RETURN gds.graph.project(
    '{projectionname}',
    p,
    null,
    {{
        sourceNodeProperties: p {{ .embedding }},
        targetNodeProperties: {{}}
    }}
    )
    """
    
    try:
        qg._graph.query(cq)
        return type, projectionname
    except():
        return False
    
def are_dictionaries_equivalent(d1, d2):
    # Check if both have the same keys
    if d1.keys() != d2.keys():
        return False

    # Check if Person1 and Person2 are swapped but equivalent
    if (d1['n1'] == d2['n2'] and d1['n2'] == d2['n1'] and d1['similarity'] == d2['similarity']):
        return True
    
    return False

def calculateSimilarityScore(projectionname: str, qg: QueryGraph):
    
    cq = f"""
    CALL gds.knn.stream('{projectionname}', {{
    topK: 1,
    nodeProperties: ['embedding'],
    // The following parameters are set to produce a deterministic result
    randomSeed: 1337,
    concurrency: 1,
    sampleRate: 1.0,
    deltaThreshold: 0.0
    }})
    YIELD node1, node2, similarity
    WHERE similarity > .90
    RETURN gds.util.asNode(node1).name AS n1, gds.util.asNode(node2).name AS n2, similarity
    ORDER BY similarity DESCENDING, n1, n2
    """

    try:
        res = qg._graph.query(cq)
    except():
        return False

    newlist = res
    
    for idx, val in enumerate(res):   
        for idx2, val2 in enumerate(newlist):
            if val != val2:
                if are_dictionaries_equivalent(val, val2):
                    newlist.pop(idx2)
    
    unique = []

    for value in newlist:
        if value not in unique:
            unique.append(value)

    return unique

def mergeNodes(type: str, listofNodes: list[str], qg: QueryGraph):
    
    for idx, val in enumerate(listofNodes):
        p1 = val['n1']
        p2 = val['n2']    
        cq = f"""
        MATCH (a1:{type} {{name: '{p1}'}}), (a2:{type} {{name: '{p2}'}})
        WITH head(collect([a1,a2])) as nodes
        CALL apoc.refactor.mergeNodes(nodes,{{
        properties:"discard",
        mergeRels:true
        }})
        YIELD node
        RETURN node;
        """
        try:   
            qg._graph.query(cq)
        except():
            return False
        
    return True

        