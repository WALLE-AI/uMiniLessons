'''
需求：论文摘要生成的Agent，如sample图片的论文解读poster
1、解析论文所有内容，分别包括内容、图片和表格
2、整体解读出
'''
import os
import threading
import httpx
from langchain_openai import ChatOpenAI
import loguru

from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class LanggrapAgentBuild():
    def __init__(self) -> None:
        self.des = "llm api service"
    def __str__(self) -> str:
        return self.des
    
    def init_client_config(self,llm_type):
        if llm_type == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
            api_key  = os.environ.get("OPENROUTER_API_KEY")
        elif llm_type =='siliconflow':
            base_url = "https://api.siliconflow.cn/v1"
            api_key = os.environ.get("SILICONFLOW_API_KEY")
        elif llm_type=="openai":
            base_url = "https://api.openai.com/v1"
            api_key  = os.environ.get("OPENAI_API_KEY")
        elif llm_type =="tongyi":
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            api_key =  os.environ.get("DASHSCOPE_API_KEY")
            
        return base_url,api_key
     
    def llm_client(self,llm_type:str,model_name:str):
        base_url,api_key = self.init_client_config(llm_type)
        thread_local = threading.local()
        try:
            return thread_local.client
        except AttributeError:
            thread_local.client = ChatOpenAI(
                model_name = model_name,
                base_url=base_url,
                api_key=api_key,
                streaming=True
                # We will set the connect timeout to be 10 seconds, and read/write
                # timeout to be 120 seconds, in case the inference server is
                # overloaded.
            )
            return thread_local.client

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]       


def test_langgraph_build_graph():
    agent = LanggrapAgentBuild()
    llm=agent.llm_client(llm_type="siliconflow",model_name="Qwen/Qwen2.5-7B-Instruct") 

    graph_builder = StateGraph(State)
    def chatbot(state: State):
        return {"messages": [llm.invoke(state["messages"])]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    graph = graph_builder.compile()
    return graph

def stream_graph_updates(user_input: str):
    graph = test_langgraph_build_graph()
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


def test_langgraph_stream():
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break


        
def test_langchain_openai_client():
    agent = LanggrapAgentBuild()
    llm=agent.llm_client(llm_type="siliconflow",model_name="Qwen/Qwen2.5-7B-Instruct")
    messages = [
    (
        "system",
        "You are a helpful translator. Translate the user sentence to chinese.",
    ),
    ("human", "I love programming."),
    ]
    respone = llm.invoke(messages)
    loguru.logger.info(f"reponse:{respone}")
        
if __name__ == "__main__":
    loguru.logger.info(f"langgraph agent build")
    test_langgraph_stream()

    
        
    
