from pymongo import MongoClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.llms import Ollama
from langchain.chains.retrieval_qa.base import RetrievalQA
import gradio as gr
from gradio.themes.base import Base

print("Hi I am here")

# Connect to MongoDB
client = MongoClient("mongodb+srv://<user>:<password>@cluster0.ufypjuf.mongodb.net/?appName=Cluster0")
collection = client["langchain_demo"]["collection_of_text_blob"]

# Embedding model (same as upload step)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector store reference
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name="vector_index",      # must match Atlas index name
    embedding_key="embedding",      # must match stored field
    text_key="text"
)

# Define query logic
def query_data(query):
    docs = vector_store.similarity_search(query, k=1)
    output = docs[0].page_content if docs else "No matching documents found."

    llm = Ollama(model="llama3")  # or mistral, phi, etc.
    retriever = vector_store.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    print((query))
    retriever_output = qa.run(query)

    return output, retriever_output


# Gradio UI
with gr.Blocks(theme=Base(), title="Ask me") as demo:
    gr.Markdown("# Question Answer App using MongoDB Vector Search + Local RAG")

    textbox = gr.Textbox(label="Enter your Question")
    button = gr.Button("Submit", variant="primary")

    output_1 = gr.Textbox(lines=4, label="Top result from MongoDB Vector Search")
    output_2 = gr.Textbox(lines=4, label="Answer from Local LLM (RAG)")

    button.click(query_data, textbox, outputs=[output_1, output_2])

demo.launch()
