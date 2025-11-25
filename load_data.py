from pymongo import MongoClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
import gradio as gr
from gradio.themes.base import Base


print("Embedding started.")

client = MongoClient("mongodb+srv://<user>:<password>@cluster0.ufypjuf.mongodb.net/?appName=Cluster0")
db_name= "langchain_demo"
collection_name = "collection_of_text_blob"
collection = client[db_name][collection_name]

loader = DirectoryLoader("./sample_files", glob="*.txt", show_progress=True)
data = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorStore = MongoDBAtlasVectorSearch.from_documents(
    data,
    embedding=embeddings,
    collection=collection,
    index_name="vector_index",        # match Atlas index
    embedding_key="embedding",        # match stored field
    text_key="text"
)

print("Embedding complete.")
