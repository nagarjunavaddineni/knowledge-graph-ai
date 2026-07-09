import streamlit as st

from graph_rag import ask_graph_rag


st.title(
    "Knowledge Graph AI Assistant"
)


person_name = st.text_input(
    "Person Name"
)


customer_name = st.text_input(
    "Customer Name"
)


question = st.text_input(
    "Ask a relationship question"
)


if st.button("Ask"):

    if (
        person_name
        and customer_name
        and question
    ):

        answer = ask_graph_rag(
            person_name,
            customer_name,
            question
        )


        st.subheader("Graph RAG Answer")

        st.write(answer)