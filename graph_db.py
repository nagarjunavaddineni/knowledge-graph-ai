import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()


NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(
        NEO4J_USERNAME,
        NEO4J_PASSWORD
    )
)


def verify_connection():

    driver.verify_connectivity()

    print(
        "Neo4j connection successful!"
    )


def run_query(
    query,
    parameters=None
):

    records, summary, keys = (
        driver.execute_query(
            query,
            parameters_=parameters or {},
            database_="neo4j"
        )
    )

    return [
        record.data()
        for record in records
    ]