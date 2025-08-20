import json
import os
import sys
import boto3
import streamlit as st
from langchain.llms.bedrock import Bedrock
from langchain.chains import RetrievalQA
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain.vectorstores import FAISS ##Vector Embeddings And Vector Store
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoaderText
from langchain.document_loaders import PyPDFLoader


##Titan Embedding to genearate Embedding

##Data Ingestion
import numpy as np

##LLM Models

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

##Bedroc Clients

bedrock=boto3.client(serviice="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1"
                                    )

## Data Ingestion
def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    # character splitter
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=10000)
    docs=text_splitter.split_documents(documents)
    return docs

##Vector embedding
def get_vector_store(docs):
    vector_store=FAISS.from_documents(docs,bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")
    
def get_claude_llm():
    llm=Bedrock(model_id="anthropic.claude-v2",
                client=bedrock  
                )
    return llm


