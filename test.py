from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)

response = llm.invoke(
    "Explain RAG in three simple sentences."
)

print(response.content)