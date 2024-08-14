from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma,FAISS
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings
from langchain_community.embeddings import OCIGenAIEmbeddings
import os
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

#In this demo we will retrieve documents and send these as a context to the LLM.

#Step 1 - setup OCI Generative AI llm

load_dotenv()
os.environ["AZURE_OPENAI_API_KEY"]=os.getenv("openai_api")
# Define the pa
# use default authN method API-key
embeddings = AzureOpenAIEmbeddings(deployment="MAJNU")

#Step 2 - here we connect to a chromadb server. we need to run the chromadb server before we connect to it

pdf_loader = PyPDFDirectoryLoader("doc" )
loaders = [pdf_loader]

documents = []
for loader in loaders:
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=100)
all_documents = text_splitter.split_documents(documents)

print(f"Total number of documents: {len(all_documents)}")



#Step 3 - since OCIGenAIEmbeddings accepts only 96 documents in one run , we will input documents in batches.

# Set the batch size
batch_size = 96

# Calculate the number of batches
num_batches = len(all_documents) // batch_size + (len(all_documents) % batch_size > 0)


texts = ["FAISS is an important library", "LangChain supports FAISS"]
db = FAISS.from_texts(texts, embeddings)
retv = db.as_retriever()

# Iterate over batches
for batch_num in range(num_batches):
    # Calculate start and end indices for the current batch
    start_index = batch_num * batch_size
    end_index = (batch_num + 1) * batch_size
    # Extract documents for the current batch
    batch_documents = all_documents[start_index:end_index]
    # Your code to process each document goes here
    retv.add_documents(batch_documents)
    print(start_index, end_index)


db.save_local("faiss_index")

