from graph_db import run_query


def find_projects(
    person_name,
    customer_name
):

    query = """
    MATCH
    (person:Person {name: $person_name})
    -[:MANAGES]->
    (project:Project)
    -[:SERVES]->
    (customer:Customer {name: $customer_name})

    RETURN
    person.name AS person,
    project.name AS project,
    project.technology AS technology,
    customer.name AS customer
    """


    return run_query(
        query,
        {
            "person_name": person_name,
            "customer_name": customer_name
        }
    )