from fastapi import APIRouter
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from app.agent.agent import agent, callbackhandler


router = APIRouter()

class ChatRequest(BaseModel):
    messages: str

@router.post('/chat')
async def chat(req: ChatRequest):
    messages = [
        SystemMessage(content = """
                            You are an enterprise assistant. Use document retrieval for knowledge questions. Do not expose internal IDs unless explicitly requested."""),
                            HumanMessage(content=req.messages)
    ]

    #response = agent.invoke(input = {'messages' : messages}, config = {'callbacks':[callbackhandler]})
    response = agent.invoke(input={'messages': messages})
    return {"answer": response['messages'][-1].content}