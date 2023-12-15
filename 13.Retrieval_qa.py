from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.callbacks import get_openai_callback
import os

os.environ['OPENAI_API_KEY'] = 'sk-'

loader = TextLoader("state_of_the_union.txt", encoding="utf-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
for i, text in enumerate(texts):
    text.metadata["source"] = f"{i}-pl"
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.chains import create_qa_with_sources_chain

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
qa_chain = create_qa_with_sources_chain(llm)
doc_prompt = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)
final_qa_chain = StuffDocumentsChain(
    llm_chain=qa_chain,
    document_variable_name="context",
    document_prompt=doc_prompt,
)
retrieval_qa = RetrievalQA(
    retriever=docsearch.as_retriever(), combine_documents_chain=final_qa_chain
)
query = "What did the president say about russia"


with get_openai_callback() as cb:
    result = retrieval_qa.run(query)
    print(cb)
    print(result)