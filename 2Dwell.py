from typing import List
from pydantic import BaseModel, Field
from langchain.chains.openai_functions import create_qa_with_structure_chain
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.chains import create_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.callbacks import get_openai_callback
from pydantic import BaseModel, Field, validator

import os

os.environ['OPENAI_API_KEY'] = 'sk-'



loader = TextLoader("files/nate.txt", encoding="utf-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
for i, text in enumerate(texts):
    text.metadata["source"] = f"{i}-pl"
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

doc_prompt = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)

class CustomResponseSchema(BaseModel):
    """An answer to the question being asked, with sources."""
    emotion: str = Field(..., description="Chose one of the emogis in the Emogi that repersent writer's feeling")
    emti: str = Field(..., description= "{chat} \n What is the chat user's personality type? among the 16 personality types."
)
    
chat = "Hey! I am sooo Happy!!!!!! I am the Best!"

  
prompt_messages = [
    SystemMessage(
        content=(
            "You are a world class algorithm to answer "
            "questions in a specific format."
        )
    ),
    HumanMessage(content="Answer question using the following context and chat"),
    HumanMessagePromptTemplate.from_template("{context}"),
    HumanMessagePromptTemplate.from_template("Question: {question}"),
]

llm = ChatOpenAI(temperature=0.8, model="gpt-3.5-turbo-0613")

chain_prompt = ChatPromptTemplate(messages=prompt_messages)

qa_chain_pydantic = create_qa_with_structure_chain(
    llm, CustomResponseSchema, output_parser="pydantic", prompt=chain_prompt
)
final_qa_chain_pydantic = StuffDocumentsChain(
    llm_chain=qa_chain_pydantic,
    document_variable_name="context",
    document_prompt=doc_prompt,
)
retrieval_qa_pydantic = RetrievalQA(
    retriever=docsearch.as_retriever(), combine_documents_chain=final_qa_chain_pydantic
)

query = "Hey! I am sooo Happy!!!!!! I am the Best! Do you know my MBTI type?"


with get_openai_callback() as cb:
    result = retrieval_qa_pydantic.run(query)
    print(cb)

    print(result)