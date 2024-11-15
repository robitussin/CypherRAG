from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

import os

# OpenAI API configuration
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key= os.environ["OPENAI_API_KEY"]
)

#Neo4j configuration
# neo4j_url = os.getenv("NEO4J_CONNECTION_URL")
# neo4j_user = os.getenv("NEO4J_USER")
# neo4j_password = os.getenv("NEO4J_PASSWORD")
neo4j_url = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "12345678"

graph = Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_password)

# cypher_generation_template_elaborated = """
# You are an expert Neo4j Cypher translator who converts English to Cypher based on the Neo4j Schema provided, following the instructions below:
# 1. Generate Cypher query compatible ONLY for Neo4j Version 5
# 2. Do not use EXISTS, SIZE, HAVING keywords in the cypher. Use alias when using the WITH keyword
# 3. Use only nodes and relationships mentioned in the schema
# 4. Always do a case-insensitive and fuzzy search for any properties related search. Eg: to search for a Person named John, use `toLower(entity.name) contains 'john'`. 
# 5. Never use relationships that are not mentioned in the given schema
# 6. When asked about entities, Match the properties using case-insensitive matching, E.g, to find a person named John, use `toLower(entity.name) contains 'john'`.
# 7. When asked about a person, Match the label property with the word "person", E.g, to find a person, use `toLower(entity.label) = 'person'`.
# 8. When asked about a place, Match the label property with the word "place", E.g, to find a place, use `toLower(entity.label) = 'place'`.
# 9. When asked about a object, Match the label property with the word "object", E.g, to find an object, use `toLower(entity.label) = 'object'`.
# 10. When asked about a event, Match the label property with the word "event", E.g, to find an event, use `toLower(entity.label) = 'event'`.
# 11. When asked about a miscellaneous entity, Match the label property with the word "miscellaneous", E.g, to find a miscellaneous entity, use `toLower(entity.label) = 'miscellaneous'`.
# 12. If a person, place, object, event or a miscellaneous entity does not match an entity in the graph, Try matching the description property or the metadata property of a relationship using case-insensitive matching, E.g, to find information about Joe, use toLower(r.description) contains 'joe' OR toLower(r.metadata) contains 'joe'.
# 13. When asked about any information of an entity, Do not simply give the entity label. Try to get the answer from the entity's relationship description or metadata property

# schema: {schema}

# Question: {question}
# """

# Cypher generation prompt
cypher_generation_template_cot = """
You are an expert Neo4j Cypher translator who converts English to Cypher based on the Neo4j Schema provided, following the instructions below:

schema: {schema}

Question: {question}

Let's break down this question into simpler, single-hop questions that will allow us to answer each part step-by-step and construct a Cypher Query:

1. Identify the main objective: What is the final piece of information that the question is asking for?
2. Determine intermediate steps: What pieces of information or entities need to be found to reach the main objective? What are the relationships between them?
3. Formulate single-hop questions: For each step, write a single-hop question that focuses on finding one specific piece of information.
4. Formulate single-hop queries: For each step, create a single-hop cypher query to retrieve each entity or relationship needed for the final answer.
5. When formulating single-hop queries, return the metadata property of the relationship between the entities
6. Construct the final query: Combine the single-hop queries into a complete Cypher query that addresses the entire question.
7. When formulating the final query, return the metadata property of the relationship between the entities

Single-Hop Questions:
1. [Insert the first single-hop question here]
2. [Insert the next single-hop question here]
3. Continue until each part of the multi-hop question is covered

Single-Hop Cypher Query:
1. [Insert the first cypher query for the first single-hop question here]
2. [Insert the next cypher query for the first single-hop question here]
3. Continue until all single-hop questions is covered

Final Cypher Query:
[Combine the single-hop queries into a complete Cypher query and insert here]
"""

# Cypher generation prompt
cypher_generation_template = """
You are an expert Neo4j Cypher translator who converts English to Cypher based on the Neo4j Schema provided, following the instructions below:
1. Generate Cypher query compatible ONLY for Neo4j Version 5
2. Do not use EXISTS, SIZE, HAVING keywords in the cypher. Use alias when using the WITH keyword
3. Use only nodes and relationships mentioned in the schema
4. Always do a case-insensitive and fuzzy search for any properties related search. Eg: to search for a Person named John, use `toLower(entity.name) contains 'john'`. 
5. Never use relationships that are not mentioned in the given schema
6. When asked about entities, Match the properties using case-insensitive matching, E.g, to find a person named John, use `toLower(entity.name) contains 'john'`.
7. If a person, place, object, event or a miscellaneous entity does not match an entity in the graph, Try matching the description property or the metadata property of a relationship using case-insensitive matching, E.g, to find information about Joe, use toLower(r.description) contains 'joe' OR toLower(r.metadata) contains 'joe'.
8. When asked about any information of an entity, Do not simply give the entity label. Try to get the answer from the entity's relationship description or metadata property
9. Use regex to help find an entity that contains multiple words, E.g, to find a the 'Notre Dame Main Building', use toLower(e.name) =~ '.*\\\\b(not|notre|dame|main|building)\\\\b.*'
10. When using MATCH traverse the relationship in both directions, E.g, (e:Entity)-[r:RELATED]-(re:Entity)

schema: {schema}
Question: {question}
"""

# Cypher generation prompt
cypher_generation_template_fewshot = """
You are an expert Neo4j Cypher translator who converts English to Cypher based on the Neo4j Schema provided, following the instructions below:
1. Generate Cypher query compatible ONLY for Neo4j Version 5
2. Do not use EXISTS, SIZE, HAVING keywords in the cypher. Use alias when using the WITH keyword
3. Use only nodes and relationships mentioned in the schema
4. Always do a case-insensitive and fuzzy search for any properties related search. Eg: to search for a Person named John, use `toLower(entity.name) contains 'john'`. 
5. Never use relationships that are not mentioned in the given schema
6. When asked about entities, Match the properties using case-insensitive matching, E.g, to find a person named John, use `toLower(entity.name) contains 'john'`.
7. If a person, place, object, event or a miscellaneous entity does not match an entity in the graph, Try matching the description property or the metadata property of a relationship using case-insensitive matching, E.g, to find information about Joe, use toLower(r.description) contains 'joe' OR toLower(r.metadata) contains 'joe'.
8. When asked about any information of an entity, Do not simply give the entity label. Try to get the answer from the entity's relationship description or metadata property
9. Use regex to help find an entity that contains multiple words, E.g, to find a the 'Notre Dame Main Building', use toLower(e.name) =~ '.*\\\\b(not|notre|dame|main|building)\\\\b.*'
10. When using MATCH traverse the relationship in both directions, E.g, (e:Entity)-[r:RELATED]-(re:Entity)

schema: {schema}

Examples:
Question: Who is John?
Answer:
MATCH (e:Entity)-[r:RELATED]-(re:Entity)
WHERE toLower(r.description) CONTAINS 'john'
OR toLower(r.metadata) CONTAINS 'john'
RETURN e.name, r.metadata, r.description, re.name

Question: Where is Manila?
Answer: 
MATCH (e:Entity)-[r:RELATED]-(re:Entity)
WHERE toLower(e.name) = 'manila'
RETURN e.name, r.metadata, r.description, re.name

Question: List all the places mentioned in the document
Answer: 
MATCH (e:Entity)
WHERE e.label = place;
RETURN e

Question: Describe the Notre Dame main building
MATCH (e:Entity)-[r:RELATED]-(re:Entity)
WHERE toLower(e.name) =~ '.*\\\\b(not|notre|dame|main|building)\\\\b.*'
RETURN e.name, r.metadata, r.description, re.name

Question: {question}
# """

cypher_prompt = PromptTemplate(
    template = cypher_generation_template,
    input_variables = ["schema", "question"]
)

cypher_prompt_cot = PromptTemplate(
    template = cypher_generation_template_cot,
    input_variables = ["schema", "question"]
)

# CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
# The information part contains the provided information that you must use to construct an answer.
# The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
# Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
# If the provided information is empty, say that you don't know the answer.
# Final answer should be easily readable and structured.
# Information:
# {context}

# Question: {question}
# Helpful Answer:"""

CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
If the provided information does not have enough information to provide a clear answer to the question, say that you don't know the answer
Provide only the direct answer without any additional explanations, context, or elaboration. For example, if asked, 'In what country is Normandy located?' your response should simply be 'France' without any additional information."
Information:
{context}

Question: {question}
Helpful Answer:"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)

class QueryGraph:
    _cypher_generation_template_cot: str
    _cypher_generation_template: str
    _cypher_prompt: PromptTemplate
    _cypher_prompt_cot: PromptTemplate
    _CYPHER_QA_TEMPLATE: str
    _qa_prompt: PromptTemplate
    _llm :ChatOpenAI
    _graph : Neo4jGraph

    def __init__(self, 
                cgt_cot: PromptTemplate = cypher_generation_template_cot,
                cgt: PromptTemplate = cypher_generation_template,
                cp: PromptTemplate = cypher_prompt,
                cpc: PromptTemplate = cypher_prompt_cot,
                cqa: str = CYPHER_QA_TEMPLATE,
                qap: PromptTemplate = qa_prompt,
                lm : ChatOpenAI = llm,
                graphdb : Neo4jGraph = graph
                ):
        
        self._cypher_generation_template_cot = cgt_cot
        self._cypher_generation_template = cgt
        self._cypher_prompt = cp
        self._cypher_prompt_cot = cpc
        self._CYPHER_QA_TEMPLATE = cqa
        self._qa_prompt = qap
        self._llm = lm
        self._graph = graphdb

    def test_query(self, user_input):
        
        prompt = PromptTemplate.from_template("""
        You are an expert Neo4j Cypher translator who converts a question in English to Cypher based on the Neo4j Schema provided below:

        schema: {schema}
        
        The question below cannot be answered directly and requires to be broken down into simpler questions:
                                            
        Question: {question}

        Let's break down this question into simpler, single-hop questions that will allow us to answer each part step-by-step and construct a Cypher Query:

        1. Identify the main objective: What is the final piece of information that the question is asking for?
        2. Determine intermediate steps: What pieces of information or entities need to be found to reach the main objective? What are the relationships between them?
        3. Formulate single-hop questions: For each step, write a single-hop question that focuses on finding one specific piece of information.
        4. Formulate single-hop queries: For each step, create a single-hop cypher query to retrieve each entity or relationship needed for the final answer.
        5. When formulating single-hop queries, return the metadata property of the relationship between the entities
        6. Construct the final query: Combine the single-hop queries into a complete Cypher query that addresses the entire question.
        7. When formulating the final query, return the metadata property of the relationship between the entities

        Single-Hop Questions:
        1. [Insert the first single-hop question here]
        2. [Insert the next single-hop question here]
        3. Continue until each part of the question is covered

        Single-Hop Cypher Query:
        1. [Insert the first cypher query for the first single-hop question here]
        2. [Insert the next cypher query for the first single-hop question here]
        3. Continue until all single-hop questions is covered

        Final Cypher Query:
        [Combine the single-hop queries into a complete Cypher query and insert here]
        """)

        chain = prompt | self._llm
        result = chain.invoke({"question": user_input, "schema": graph.schema})
        return result

    def query_graph_cot(self, user_input):
        chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
            return_intermediate_steps=True,
            cypher_prompt=self._cypher_prompt_cot,
            qa_prompt=self._qa_prompt)
        try:
            result = chain.invoke(user_input)
            return result
        except Exception as e:
            print(e)
            return None
        
    def runquestion(self, user_input):
        chain = GraphCypherQAChain.from_llm(
            llm=self._llm,
            graph=self._graph,
            verbose=True,
            return_intermediate_steps=True,
            cypher_prompt=self._cypher_prompt,
            qa_prompt=self._qa_prompt)
        
        try:
            result = chain.invoke(user_input)
            return result
        except Exception as e:
            print(e)
            return None

    def refine_query(self, previous_query, user_input):

        cypher_refine_template = f"""" 
        Context: I am working with a Neo4j database containing information.

        Initial Cypher Query:
            {previous_query}
        """

        cypher_refine_template += """
        Problem:
        The above Cypher Query returned no results. 
        I need to refine this query to achieve to answer the question "{question}": 

        Schema: {schema}

        Request:
        Can you please refine the Initial Cypher Query to answer the question?
        """
            
        cypher_refine_prompt = PromptTemplate(input_variables=["schema", "question"], template=cypher_refine_template)
  
        chain = GraphCypherQAChain.from_llm(llm=self._llm, 
                                            graph=self._graph, 
                                            verbose=False, 
                                            return_intermediate_steps=True, 
                                            cypher_prompt=cypher_refine_prompt)
        
        try:
                result = chain.invoke(user_input)
                return result
        except Exception as e:
                print(e)
                return None

        #print(cypher_refine_prompt.format(question=user_input, schema=graph.schema, prevquery=previous_query))

    def hello(self):
         print("hello")
         