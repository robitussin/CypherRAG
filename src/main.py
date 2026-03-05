import streamlit as st
from streamlit_chat import message
from timeit import default_timer as timer
from query_graph import QueryGraph

qg = QueryGraph()

st.set_page_config(layout="wide")

if "user_msgs" not in st.session_state:
    st.session_state.user_msgs = []
if "system_msgs" not in st.session_state:
    st.session_state.system_msgs = []

title_col, empty_col, img_col = st.columns([2, 1, 2])    

with title_col:
    st.title("Conversational Neo4J Assistant")
with img_col:
    st.image("https://dist.neo4j.com/wp-content/uploads/20210423062553/neo4j-social-share-21.png", width=200)

user_input = st.text_input("Enter your question", key="input")
if user_input:
    with st.spinner("Processing your question..."):
        st.session_state.user_msgs.append(user_input)
        start = timer()

        print("user input", user_input)
        try:
            requirements = qg.get_requirements(user_input)
            result = qg.answer_question(user_input, requirements)
            print("result: ", result)
            intermediate_steps = result["intermediate_steps"]
            cypher_query = intermediate_steps[0]["query"]
            database_results = intermediate_steps[1]["context"]
            answer = result["result"]   

            if answer == "I don't know the answer.":
                result = qg.refine_query(cypher_query[6:], user_input)
                intermediate_steps = result["intermediate_steps"]
                cypher_query = intermediate_steps[0]["query"]
                database_results = intermediate_steps[1]["context"]
                answer = result["result"]   
            
            st.session_state.system_msgs.append(answer)
            # else:
            #     st.session_state.system_msgs.append(answer)
        except Exception as e:
            st.write("Failed to process question. Please try again.")
            print("error: ", e)
            # print(e)

    st.write(f"Time taken: {timer() - start:.2f}s")

    col1, col2, col3 = st.columns([1, 1, 1])

    # Display the chat history
    with col1:
        if st.session_state["system_msgs"]:
            for i in range(len(st.session_state["system_msgs"]) - 1, -1, -1):
                message(st.session_state["system_msgs"][i], key = str(i) + "_assistant")
                message(st.session_state["user_msgs"][i], is_user=True, key=str(i) + "_user")

    with col2:
        if cypher_query:
            st.text_area("Last Cypher Query", cypher_query, key="_cypher", height=240)
        
    with col3:
        if database_results:
            st.text_area("Last Database Results", database_results, key="_database", height=240)
    