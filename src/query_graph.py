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


cypher_generation_template = """

    You are a Cypher query generator for a Neo4j graph database. Your task is to translate user questions into precise Cypher queries. Handle both simple and multi-hop questions by following these steps:
    
    This is the question: {question}
    These are the requirements for the question: {requirements}  
    
    Follow the steps to generate a cypher query using the requirements provided.
    1. From the requirements, generate a cypher query following the given format below.
    2. Do not use node labels.
    3. Use generic node variables (e.g., w, x, y).
    4. Define relationships explicitly (e.g., [r1], [r2]).
    5. Include filters for metadata or descriptions using toLower() for case-insensitivity.
    6. Use AND or OR in WHERE clauses to combine conditions.
    7. Return DISTINCT results to avoid duplicates.
    8. The variables in RETURN must answer satisfy the requirements.
    9. Use UNION ALL if the question requires to compare two entities.
    10. When using UNION ALL, always use the same alias for both MATCH clauses.

    Format for simple questions:
    MATCH (w)-[r1]-(x)
    WHERE (
        toLower(r1.metadata) =~  '^.*\\\\b(keyentity)\\\\w*\\\\b.*$' OR
        toLower(r1.description) =~  '^.*\\\\b(keyentity)\\\\w*\\\\b.*$'
    )
    RETURN DISTINCT r1.metadata, r1.description
    
    Format for multi-hop questions:
    MATCH (w)-[r1]-(x)-[r2]-(y)
    WHERE (
        toLower(r1.metadata) =~ '^.*\\\\b(keyentity)\\\\w*\\\\b.*$' OR
        toLower(r1.description) =~ '^.*\\\\b(keyentity)\\\\w*\\\\b.*$'
    )
    AND (
        toLower(r2.metadata) =~ '^.*\\\\b(keyentity)\\\\w*\\\\b.*$' OR
        toLower(r2.description) =~ '^.*\\\\b(keyentity)\\\\w*\\\\b.*$'
    )
    RETURN DISTINCT r1.metadata,r1.description,r2.metadata,r2.description
 
    Format for comparison questions:
    MATCH ()-[r1]-()
    WHERE (
        toLower(r1.metadata) =~ '^.*\\\\b(keyentity)\\\\w*\\\\b.*$' OR
        toLower(r1.description) =~ '^.*\\\\b(keyentity)\\\\w*\\\\b.*$'
    )
    RETURN DISTINCT r1.metadata AS col1
    UNION ALL
    MATCH ()-[r2]-()
    WHERE (
        toLower(r2.metadata) =~ '^.*\\\\b(keyentity)\\\\w*\\\\b.*$' OR 
        toLower(r2.description) =~ '^.*\\\\b(keyentity)\\\\w*\\\\b.*$'
    )
    RETURN DISTINCT r2.metadata AS col1

    Examples:
    Question: Where is Manila?
    Answer: 
    MATCH (w)-[r1]-(x)
    WHERE (
        toLower(r1.metadata) =~ '^.*\\\\b(manila)\\\\w*\\\\b.*$'OR
        toLower(r1.description) =~ '^.*\\\\b(manila)\\\\w*\\\\b.*$'
    )
    RETURN r1.metadata, r1.description 
    
    User Question: "What government position was held by the man who portrayed Jack Browning in the film The Killers?"
    MATCH (w)-[r1]-(x)-[r2]-(y)
    WHERE (
        toLower(r1.metadata) =~ '^.*\\\\b(jack browning)\\\\w*\\\\b.*$' OR
        toLower(r1.description) =~ '^.*\\\\b(the killers)\\\\w*\\\\b.*$'
    )
    AND (
        toLower(r2.metadata) =~ '^.*\\\\b(government|position)\\\\w*\\\\b.*$'
        toLower(r2.description) =~ '^.*\\\\b(government|position)\\\\w*\\\\b.*$'
    )
    RETURN DISTINCT r1.metadata,r1.description,r2.metadata,r2.description

    User Question: "The director of the romantic comedy Friends with Benefits is based in what New York city?"
    MATCH (w)-[r1]-(x)-[r2]-(y)
    WHERE (
        toLower(r1.metadata) =~ '^.*\\\\b(friends with benefits)\\\\w*\\\\b.*$' OR
        toLower(r1.description) =~ '^.*\\\\b(director)\\\\w*\\\\b.*$'
    )
    AND (
        toLower(r2.metadata) =~ '^.*\\\\b(new york)\\\\w*\\\\b.*$' OR
        toLower(r2.description) =~ '^.*\\\\b(new york)\\\\w*\\\\b.*$'
    )
    RETURN DISTINCT r1.metadata,r1.description,r2.metadata,r2.description

    User Question: "Are Lebron James and Jayson Tatum both basketball players??"
    MATCH ()-[r1]-()
    WHERE (
        toLower(r1.metadata) =~ '^.*\\\\b(lebron james)\\\\w*\\\\b.*$' OR
        toLower(r1.description) =~ '^.*\\\\b(lebron james)\\\\w*\\\\b.*$'
    )
    RETURN DISTINCT r1.metadata AS info1
    UNION ALL
    MATCH ()-[r2]-()
    WHERE (
        toLower(r2.metadata) =~ '^.*\\\\b(jayson tatum)\\\\w*\\\\b.*$' OR 
        toLower(r2.description) =~ '^.*\\\\b(jayson tatum)\\\\w*\\\\b.*$'
    )
    RETURN DISTINCT r2.metadata AS info1
    """

cypher_prompt = PromptTemplate(
    template = cypher_generation_template,
    input_variables = ["question", "requirements"]
)

CYPHER_QA_TEMPLATE = """
You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
If the provided information is empty, say that you don't know the answer.
Provide only the direct answer without any additional explanations, context, or elaboration. For example, if asked, 'In what country is Normandy located?' your response should simply be 'France' without any additional information."

Information: {context}
Question: {question}
Helpful Answer:
"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)

class QueryGraph:
    _cypher_prompt: PromptTemplate
    _CYPHER_QA_TEMPLATE: str
    _qa_prompt: PromptTemplate
    _llm : ChatOpenAI
    _graph : Neo4jGraph

    def __init__(self, 
                cp: PromptTemplate = cypher_prompt,
                cqa: str = CYPHER_QA_TEMPLATE,
                qap: PromptTemplate = qa_prompt,
                lm : ChatOpenAI = llm,
                graphdb : Neo4jGraph = graph
                ):
        
        self._cypher_prompt = cp
        self._CYPHER_QA_TEMPLATE = cqa
        self._qa_prompt = qap
        self._llm = lm
        self._graph = graphdb

    def test_prompt(self, user_input):
                
        prompt = PromptTemplate.from_template("""
            
            What is required from this question? 
                                              
            Question: {question}
                                              
            List the requirements here
            [requirements]
                                              
            From the requirements, generate a cypher query
            Cypher Query:
            # """)

        chain = prompt | self._llm
        result = chain.invoke({"question": user_input, "schema": graph.schema})
        return result
    
    
    def get_requirements(self, user_input):

        prompt = PromptTemplate.from_template("""
                                                    
        You are an intelligent system that first analyzes the structure and requirements of a question, and then uses those insights to identify the relevant components needed to answer the question. You will then provide a list of requirements that are necessary to answer the question, before querying any databases.

        Please perform the following steps:

        1. Identify the key components of the question. What is the user asking for? What are the main aspects or topics involved? For example, the question may ask for a specific type of product, event, or characteristic.
        2. Based on your understanding of the question, list the requirements needed to answer it. Requirements could include things like:
        - The type of information or entities needed (e.g., book title, author, genre).
        - Any conditions or constraints that must be met (e.g., "first-person narrative," "young adult audience").
        - Context or additional information that will influence how the question is answered (e.g., related entities or categories, such as "sci-fi genre," or "companion books").
        3. If any additional context or data is needed to answer the question, mention that here.
        4. Do not provide any extra information except the list 
        5. Do not provide any requirement to search information from other sources.

        Question: {question}
        """)

        chain = prompt | self._llm
        result = chain.invoke({"question": user_input})
        return result
  
    def answer_question(self, user_input, req):
        chain = GraphCypherQAChain.from_llm(
            llm=self._llm,
            graph=self._graph,
            verbose=True,
            return_intermediate_steps=True,
            cypher_prompt=self._cypher_prompt,
            qa_prompt=self._qa_prompt,
            top_k=50)
        try:
            # result = chain.invoke(user_input)
            result = chain.invoke(input={"query":user_input, "requirements":req})
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
         