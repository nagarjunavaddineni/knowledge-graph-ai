import json

from graph_db import (
    run_query,
    verify_connection
)


DATA_FILE = "data/company_data.json"


def load_data():

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def build_graph():

    data = load_data()


    for person in data["people"]:

        run_query(
            """
            MERGE (p:Person {name: $name})
            SET p.role = $role
            """,
            person
        )


    for project in data["projects"]:

        run_query(
            """
            MERGE (p:Project {name: $name})
            SET p.technology = $technology
            """,
            project
        )


    for customer in data["customers"]:

        run_query(
            """
            MERGE (c:Customer {name: $name})
            SET c.industry = $industry
            """,
            customer
        )


    for relationship in data["relationships"]:

        run_query(
            """
            MATCH (person:Person {
                name: $person
            })

            MATCH (project:Project {
                name: $project
            })

            MERGE (person)-[:MANAGES]->(project)
            """,
            relationship
        )


    for relationship in data[
        "project_customers"
    ]:

        run_query(
            """
            MATCH (project:Project {
                name: $project
            })

            MATCH (customer:Customer {
                name: $customer
            })

            MERGE (project)-[:SERVES]->(customer)
            """,
            relationship
        )


    print(
        "Knowledge graph created successfully!"
    )


if __name__ == "__main__":

    verify_connection()

    build_graph()