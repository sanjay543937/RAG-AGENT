
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="employee_policies",
    persist_directory="chroma_db",
    embedding_function=embeddings,
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

@tool
def search_employee_policies(query: str) -> str:
    """Search the employee policy documents for relevant information."""
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant policy information was found."

    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n"
        f"Content: {doc.page_content}"
        for doc in docs
    )

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0,
)

agent = create_agent(
    model=llm,
    tools=[search_employee_policies],
    system_prompt = """
You are a friendly employee helpdesk assistant.

Rules:
1. If the user greets you, such as "Hi", "Hello",
   or "Good morning", respond politely without calling any tool.
2. For questions about employee policies, use the
   search_employee_policies tool.
3. Answer policy questions using the retrieved documents.
4. If the documents do not contain the answer, say you
   do not have enough information.
5. Never invent company policies.
6. Do not call tools for general greetings or small talk.
"""
)

def main():
    print("Employee Helpdesk RAG Agent")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        try:
            result = agent.invoke({
                "messages": [{"role": "user", "content": question}]
            })

            print("\nAssistant:", result["messages"][-1].content, "\n")

        except Exception as error:
            print(f"Error: {error}")

if __name__ == "__main__":
    main()