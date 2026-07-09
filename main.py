from graph_rag import ask_graph_rag


print("=" * 60)

print("KNOWLEDGE GRAPH AI ASSISTANT")

print("=" * 60)


person_name = input(
    "\nEnter person name: "
)


customer_name = input(
    "Enter customer name: "
)


question = input(
    "Ask your question: "
)


answer = ask_graph_rag(
    person_name,
    customer_name,
    question
)


print("\nAssistant:")

print(answer)