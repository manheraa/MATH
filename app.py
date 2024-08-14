from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings
from langchain_community.embeddings import OCIGenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
import os

#In this demo we will retrieve documents and send these as a context to the LLM.

#Step 1 - setup OCI Generative AI llm


# use default authN method API-key
os.environ["AZURE_OPENAI_API_KEY"]=os.getenv('openai_api')
embeddings=AzureOpenAIEmbeddings(deployment='MAJNU')

#Step 2 - here we connect to a chromadb server. we need to run the chromadb server before we connect to it

#Step 3 - here we crete embeddings using 'cohere.embed-english-light-v2.0" model.

llm=AzureChatOpenAI(azure_deployment='loloa',
            api_version="2024-05-01-preview",
            )
#Step 4 - here we create a retriever that gets relevant documents (similar in meaning to a query)

db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

#retv = db.as_retriever(search_kwargs={"k": 3})

retv = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

#Step 5 - here we can explore how similar documents to the query are returned by prining the document metadata. This step is optional

docs = retv.invoke('what is Laplace rule')

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

pretty_print_docs(docs)

docs1 = []

for doc in docs:
    #print(doc)
    docs1.append({"snippet": doc.page_content})
    #print(docs1)


#Step 4 - here we create a retrieval chain that takes llm , retirever objects and invoke it to get a response to our query

chain = RetrievalQA.from_chain_type(llm=llm, retriever=retv,return_source_documents=True)
import streamlit as st
st.title("Math Bot")
x=st.text_input("Enter The question")
if x:
    response = chain.invoke(x)
    st.write(response)
# ChatOCIGenAI supports documents, following code passes the documents directly to ChatOCIGenAI directly.
#messages = [HumanMessage(content="Tell us which module of AI Foundations course is relevant to Transformers")]
#response = llm.invoke(messages,documents=docs1)

