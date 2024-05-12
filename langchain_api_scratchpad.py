from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

config = dotenv_values(".env")

llm = ChatOpenAI(api_key=config["OPEN_AI_KEY"], model='gpt-4-turbo')

llm.invoke("How can LangSmith help with testing")

# LLM Chains
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technology technical documentation writer with vast experience in modern big-data architecture."),
    ("user", "{input}")
])

chain = prompt | llm

print(chain.invoke({"input": """Give me some important features of such a metadata driven architecture:
The Azure Data Ingestor is a metadata-driven architecture used to automate data ingestion pipelines for structured files, perform configurable Data Quality validations and dynamically ingest to multiple targets. 
"""}))
