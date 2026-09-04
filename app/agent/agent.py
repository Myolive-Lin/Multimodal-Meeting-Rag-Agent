import os
from typing import Annotated, TypedDict
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.graph import START, StateGraph
from app.tool.rag_tools import query_documents, read_audio, read_image, read_video, ingest_media
from app.config import settings
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler


os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
os.environ["LANGFUSE_HOST"] = settings.langfuse_host
callbackhandler = CallbackHandler()


tools = [query_documents, read_audio, read_image, read_video, ingest_media]

llm = ChatOpenAI(
    model=settings.llm_model,
    temperature=0.1,
    base_url=settings.model_url,
    api_key=settings.token_api_key
)

chat_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def assistant(agent_state: AgentState):
    return {
        'messages': [chat_with_tools.invoke(agent_state['messages'])]
    }


builder = StateGraph(AgentState)
builder.add_node("assistant", assistant )
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition
)
builder.add_edge('tools', 'assistant')


agent = builder.compile()
