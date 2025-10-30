from dotenv import dotenv_values
import os
from openai import OpenAI

config = dotenv_values(".env")

client = OpenAI(api_key=config["OPEN_AI_KEY"])

# Configure embeddings model
from langchain_openai import OpenAIEmbeddings

model_name = 'text-embedding-3-large'
embeddings = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=config["OPEN_AI_KEY"]
)

# Configure Pinecone
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=config["PINECONE_API_KEY"])
index = pc.Index(host="https://test-index-3-kuv1rfi.svc.aped-4627-b74a.pinecone.io")

# Method to create embeddings
from openai import OpenAI

def get_embedding(text, model="text-embedding-3-large"):
    """
    Generates a vector embedding for the given text using OpenAI's model.
    """
    text = text.replace("\n", " ") # Best practice is to replace newlines
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    # The embedding is in the 'data' array of the response
    return response.data[0].embedding

# # Example usage:
# sentence = "What does the knowledge source say about the speed of light?"
# embedding_vector = get_embedding(sentence)
#
# print(f"Original sentence: {sentence}")
# print(f"Embedding vector (first 50 dimensions): {embedding_vector[:50]}...")
# print(f"Vector dimension: {len(embedding_vector)}")

# Initialize OpenAI chat model
import langchain
langchain.verbose = False
# langchain.debug = False
# langchain.llm_cache = False
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4.1-mini", model_provider="openai", api_key=config["OPEN_AI_KEY"])

# v2

from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent

# Create dynamic prompt
@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text

    # Get embedding for message
    embedding = get_embedding(last_query)

    # Perform search against Pinecone
    response = index.query(
        namespace="__default__",
        vector=embedding,
        top_k=3,
        include_metadata=True,
        include_values=False
    )

    response = response.to_dict()

    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{response['matches']}"
    )

    return system_message


agent = create_agent(model, tools=[], middleware=[prompt_with_context])

# Execute
query = "What does the knowledge source say about Maka Projects?"
while True:
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()


