from openai import OpenAI
from dotenv import load_dotenv

from graph_search import find_projects


load_dotenv()


client = OpenAI()


def ask_graph_rag(
    person_name,
    customer_name,
    question
):

    graph_results = find_projects(
        person_name,
        customer_name
    )


    if not graph_results:

        return (
            "I could not find a matching "
            "relationship in the knowledge graph."
        )


    graph_context = "\n".join(
        [
            (
                f"{item['person']} manages "
                f"{item['project']}. "
                f"The project uses "
                f"{item['technology']} and serves "
                f"{item['customer']}."
            )
            for item in graph_results
        ]
    )


    response = client.responses.create(
        model="gpt-5.4-mini",
        instructions="""
        You are an enterprise knowledge graph
        assistant.

        Answer using ONLY the graph context.

        Explain the relationship path clearly.

        Do not invent people, projects, customers,
        technologies, or relationships.

        If the graph context does not support a claim,
        do not make that claim.
        """,
        input=f"""
        GRAPH CONTEXT:

        {graph_context}

        USER QUESTION:

        {question}
        """
    )


    return response.output_text