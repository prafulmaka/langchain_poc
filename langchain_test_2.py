from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor



config = dotenv_values(".env")

llm = ChatOpenAI(api_key=config["OPEN_AI_KEY"], model="gpt-3.5-turbo-0125", temperature=0)

# Create tools for:
# 1. Reading from PDF using user specified path
# 2. LangChain docs to be used when user wants to modify the sourcecode + add features

# Create tool for reading pdf
@tool
def hubspot_docs_tool(user_input: str) -> str:
    """
    This tool is used to get HubSpot documentation and answer user questions specific to HubSpot.
    """

    # Load from webpage
    loader = WebBaseLoader("https://developers.hubspot.com/docs/api/overview")

    docs = loader.load()

    # Create embeddings
    embeddings = OpenAIEmbeddings(api_key=config["OPEN_AI_KEY"])
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)

    # Create retriever
    retriever = vector.as_retriever()
    hubspot_docs_retriever = create_retriever_tool(retriever, "hubspot_docs_tool", "This tool is used to answer questions specific to the HubSpot documentation.")
    response = hubspot_docs_retriever.invoke(user_input)
    return response

@tool
def langchain_docs_tool(user_input: str) -> str:
    """
    This tool is used to get LangChain documentation and answer user questions specific to LangChain.
    """

    # Load from webpage
    loader = WebBaseLoader("https://python.langchain.com/v0.1/docs/get_started/introduction/")

    docs = loader.load()

    # Create embeddings
    embeddings = OpenAIEmbeddings(api_key=config["OPEN_AI_KEY"])
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)

    # Create retriever
    retriever = vector.as_retriever()
    langchain_docs_retriever = create_retriever_tool(retriever, "langchain_docs_tool", "This tool is used to answer questions specific to the LangChain documentation.")
    response = langchain_docs_retriever.invoke(user_input)
    return response

tools = [hubspot_docs_tool, langchain_docs_tool]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
# prompt.messages

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# User question
user_input = ""
while user_input != "XXX":
    user_input = input("Enter your question: ")
    print(agent_executor.invoke({"input": user_input}).get('output'))
