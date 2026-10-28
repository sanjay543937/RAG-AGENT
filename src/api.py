
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import agent

app = FastAPI(
    title="Employee Helpdesk RAG API",
    description="REST API for answering employee policy questions",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="The employee's question",
    )


class ChatResponse(BaseModel):
    answer: str


@app.get("/")
def home():
    return {
        "message": "Employee Helpdesk RAG API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": question}
            ]
        })

        answer = result["messages"][-1].content

        return ChatResponse(answer=answer)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate an answer: {error}",
        ) from error