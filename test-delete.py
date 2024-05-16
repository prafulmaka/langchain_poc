from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain.tools.retriever import create_retriever_tool



config = dotenv_values(".env")

llm = ChatOpenAI(api_key=config["OPEN_AI_KEY"])

# Create tools for:
# 1. Reading from PDF using user specified path
# 2. LangChain docs to be used when user wants to modify the sourcecode + add features


"""This tool is used to get documentation from the PDF document shared by the user."""
pdf_path_input = input("Enter file path: ")
print(f"Fetching details from: {pdf_path_input}")

# Load PDF
loader = PyPDFLoader(pdf_path_input)

docs = loader.load()

# Create embeddings
embeddings = OpenAIEmbeddings(api_key=config["OPEN_AI_KEY"])
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# Create retrieval chain
retriever = vector.as_retriever()

retriever_tool = create_retriever_tool(retriever, "helpful_agent", "helps understand documentation")

tools = [retriever_tool]

from langchain import hub

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages

from langchain.agents import create_tool_calling_agent

agent = create_tool_calling_agent(llm, tools, prompt)

from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "What is INFER?"})

