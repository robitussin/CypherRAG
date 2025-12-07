import os
from knowledge_graph_maker import GraphMaker, Ontology, OpenAIClient, Edge
from knowledge_graph_maker import Document
import datetime

from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel
from typing import Optional, List
from knowledge_graph_maker import Neo4jGraphModel

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = "TEST"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "12345678"
os.environ["NEO4J_URI"]= "bolt://localhost:7687"

# Pydantic data class
class Sentences(BaseModel):
    sentences: List[str]

def get_propositions(text: str, proposition_list: List[str]): 
    client = Client()
    obj = client.pull_prompt("wfh/proposal-indexing")

    chunking_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # use it in a runnable
    runnable = obj | chunking_llm

    # Extraction
    structured_llm = chunking_llm.with_structured_output(Sentences)

    # text = text.split(".")

    file = open("propositions.txt", "a")

    for t in text:
        
        if t:
            print("t:",t)
            runnable_output = runnable.invoke({
                "input": t
            }).content
        
            propositions = structured_llm.invoke(runnable_output).sentences
            
            for item in propositions:
                file.write(item + "\n")
            # proposition_list.extend(propositions)
            
    file.close()       
    # return proposition_list
    
def get_propositions_nosplit(text: str, proposition_list: List[str]):
    client = Client()
    obj = client.pull_prompt("wfh/proposal-indexing")
    
    chunking_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # use it in a runnable
    runnable = obj | chunking_llm

    # Extraction
    structured_llm = chunking_llm.with_structured_output(Sentences)

    file = open("propositions.txt", "a")
    
    if text:
        runnable_output = runnable.invoke({
            "input": text
        }).content
    
        propositions = structured_llm.invoke(runnable_output).sentences
        
        for item in propositions:
            file.write(item + "\n")
        # proposition_list.extend(propositions)
            
    file.close()       
    # return proposition_list


def generate_summary(text: str, llm: OpenAIClient):
    SYS_PROMPT = (
        "Succintly summarise the text provided by the user. "
        "Respond only with the summary and no other comments"
    )
    try:
        summary = llm.generate(user_message=text, system_message=SYS_PROMPT)
    except:
        summary = ""
    finally:
        return summary

def generateEdges(proposition_list: List[str]) -> List[Edge]:

    # ontology = Ontology(
    #     labels=[
    #         {"University": "The overall institution that governs all campuses, colleges, and departments."},
    #         {"Campus": "A physical site of the university, typically with its own facilities and offices."},
    #         {"College": "An academic unit within the university overseeing multiple departments."},
    #         {"Department": "A division within a college focused on a specific academic discipline or administrative function."},
    #         {"Person": "An individual such as a faculty member, staff, administrator, or student."},
    #         {"Position": "A designated role or office held by a person within the institution’s structure."},
    #         {"Committee": "A group formed to perform specific governance, academic, or administrative functions."},
    #         {"Policy": "A documented rule, guideline, or procedure governing academic or administrative matters."},
    #         {"Rank": "An academic or administrative level assigned to a person (e.g., Assistant Professor, Dean)."},
    #         {"Course": "An instructional unit offered by a department or college."},
    #         {"Activity": "An academic or administrative task, event, or project performed within the institution."},
    #         {"LeaveType": "A form of authorized absence from duty (e.g., sabbatical, sick leave)."},
    #         {"Publication": "A scholarly or creative work authored by faculty or staff."},
    #     ],

    #          relationships=[
    #             "University HAS_COLLEGE College",
    #             "College HAS_DEPARTMENT Department",
    #             "Department WORKS_IN Person",
    #             "Person HAS_ROLE Position",
    #             "Position REPORTS_TO Position",
    #             "Person MEMBER_OF Committee",
    #             "Committee PART_OF Department",
    #             "Department HAS_POLICY Policy",
    #             "Policy MENTIONS Department",
    #             "Department LOCATED_AT Campus",
    #             "Person SUPERVISES Person",
    #             "Person REPLACES Person",
    #             "Person TEACHES Course",
    #             "Department OFFERS Course",
    #             "Person PUBLISHED Publication"
    #     ],
        
        # relationships=[
        #     "University HAS_COLLEGE College",
        #     "College HAS_DEPARTMENT Department",
        #     "Department WORKS_IN Person",
        #     "Person HAS_ROLE Position",
        #     "Position REPORTS_TO Position",
        #     "Person MEMBER_OF Committee",
        #     "Committee PART_OF Department",
        #     "Department HAS_POLICY Policy",
        #     "Policy MENTIONS Department",
        #     "Department LOCATED_AT Campus",
        #     "Person SUPERVISES Person",
        #     "Person REPLACES Person",
        #     "Person TEACHES Course",
        #     "Department OFFERS Course",
        #     "Person PUBLISHED Publication"
        # ],
    # )

    ontology = Ontology(
   labels=[
        {"Person": "Named individual (can be referenced by name or pronoun)."},
        {"Organization": "Corporation, institution, agency, or association."},
        {"Place": "Geographic or physical location where events or entities exist."},
        {"Event": "Something that happens at a specific time and/or place."},
        {"Object": "Inanimate entity used by persons or found in places."},
        {"Miscellaneous": "Important concept not covered by the above labels."},
    ],
    relationships=[
        "Relation between any pair of Entities"
        ],
    )

    ## Open AI models
    oai_model="gpt-4o-mini"

    ## OR Use OpenAI
    llm = OpenAIClient(model=oai_model, temperature=0.1, top_p=0.5)

    current_time = str(datetime.datetime.now())

    docs = map(lambda t: Document(text=t, metadata={"summary": t, 'generated_at': current_time}),proposition_list)
    
    graph_maker = GraphMaker(ontology=ontology, llm_client=llm, verbose=True)

    graph = graph_maker.from_documents(list(docs), delay_s_between=0)
    
    return graph

def createGraph(graph: List[Edge]) -> bool:
    
    create_indices = False
    neo4j_graph = Neo4jGraphModel(edges=graph, create_indices=create_indices)
    neo4j_graph.save()

    return True
    

    