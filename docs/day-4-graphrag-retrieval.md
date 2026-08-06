# Day 4: Advanced GraphRAG Retrieval

Day 4 adds a production-oriented GraphRAG retrieval and answer-generation
pipeline to the Knowledge Graph AI project.

## Architecture

```text
User question
    |
    +--> Full-text chunk retrieval
    |
    +--> Semantic vector retrieval
              |
              v
     Reciprocal rank fusion
              |
              v
      Graph-context expansion
              |
              v
       Grounded LLM answer
              |
              v
      Sources and graph facts