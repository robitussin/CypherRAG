total nodes

MATCH (n)
RETURN count(n) AS total_nodes;

total relationships

MATCH ()-[r]->()
RETURN count(r) AS total_relationships;

node distribution

MATCH (n)
UNWIND labels(n) AS label
RETURN label, count(*) AS count
ORDER BY count DESC;

relationship distribution type

MATCH ()-[r]->()
RETURN type(r) AS relationship_type, count(*) AS count
ORDER BY count DESC;

avg degree of graph
MATCH (n)
WITH count(n) AS total_nodes
MATCH ()-[r]->()
WITH total_nodes, count(r) AS total_relationships
RETURN (2.0 * total_relationships) / total_nodes AS avg_degree;

node degree distribution
MATCH (n)
RETURN size((n)--()) AS degree, count(*) AS num_nodes
ORDER BY degree DESC;

graph density
MATCH (n)
WITH count(n) AS total_nodes
MATCH ()-[r]->()
WITH total_nodes, count(r) AS total_relationships
RETURN 
total_relationships * 1.0 / (total_nodes * (total_nodes - 1)) AS density;

detect node duplicates
MATCH (n)
WITH n.name AS name, collect(n) AS nodes
WHERE size(nodes) > 1
RETURN name, size(nodes) AS occurrences
ORDER BY occurrences DESC;

fragmentation
CALL gds.wcc.stats('yourGraphProjection')
YIELD componentCount, nodeCount, relationshipCount;

largest connected component
CALL gds.wcc.stream('yourGraphProjection')
YIELD componentId, nodeId
RETURN componentId, count(*) AS size
ORDER BY size DESC
LIMIT 5;