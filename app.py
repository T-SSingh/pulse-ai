from dotenv import load_dotenv
load_dotenv(override=True)
import os
import chainlit as cl

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
embeddings = OpenAIEmbeddings()
client = QdrantClient(
    url=os.environ["QDRANT_CLOUD_URL"], 
    api_key=os.environ["QDRANT_API_KEY"], 
)
qdrant = Qdrant( client=client, collection_name="exhibitors", embeddings=embeddings)
qdrant_retriever = qdrant.as_retriever()

from langchain_core.prompts import ChatPromptTemplate
RAG_PROMPT = """
CONTEXT:
{context}

QUERY:
{question}

You are a helpful assistant. Use the available context to answer the question. If you can't answer the question, say you don't know.
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
from langchain_openai import ChatOpenAI
openai_chat_model = ChatOpenAI(model="gpt-3.5-turbo")
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
rag_chain = (
    {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
    | rag_prompt | openai_chat_model | StrOutputParser()
)
from typing import Annotated
from langchain_core.tools import tool

@tool
def retrieve_exhibitor_information(
    query: Annotated[str, "query to ask the retrieve exhibitors information"]
    ):
  """Use Retrieval Augmented Generation to retrieve information about exhibitors."""
  return rag_chain.invoke({"question" : query})


tools = [TavilySearchResults(max_results=1), retrieve_exhibitor_information]

from langgraph.prebuilt import ToolExecutor

tool_executor = ToolExecutor(tools)

from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0, streaming=True)

from langchain.tools.render import format_tool_to_openai_function

functions = [format_tool_to_openai_function(t) for t in tools]
model = model.bind_functions(functions)

from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import FunctionMessage


def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]   
    if "function_call" not in last_message.additional_kwargs:
        return "end"  
    else:
        return "continue"


def call_model(state):
    messages = state['messages']
    response = model.invoke(messages)  
    return {"messages": [response]}


def call_tool(state):
    messages = state['messages'] 
    last_message = messages[-1]  
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
    )  
    response = tool_executor.invoke(action) 
    function_message = FunctionMessage(content=str(response), name=action.tool)   
    return {"messages": [function_message]}

from langgraph.graph import StateGraph, END
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

workflow.set_entry_point("agent")


workflow.add_conditional_edges(
    "agent", 
    should_continue,
    {        
        "continue": "action",      
        "end": END
    }
)

workflow.add_edge('action', 'agent')

app = workflow.compile()

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

@cl.on_message
async def run_convo(message: cl.Message):   
    msg = cl.Message(content="")
    await msg.send()  
    await cl.sleep(1) #hack to simulate loader!

    inputs = {"messages": [HumanMessage(content=message.content)]}

    res = app.invoke(inputs, config=RunnableConfig(callbacks=[
        cl.LangchainCallbackHandler(
            to_ignore=["ChannelRead", "RunnableLambda", "ChannelWrite", "__start__", "_execute"]      
        )]))

    await cl.Message(content=res["messages"][-1].content).send() 

   