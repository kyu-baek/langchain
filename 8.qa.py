from langchain import OpenAI
from langchain.chains.question_answering import load_qa_chain

llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0)
qa_chain = load_qa_chain(llm, chain_type="map_reduce")

from langchain.chains import AnalyzeDocumentChain

qa_doc_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)

with open('chat.txt') as f:
    chat = f.read()

print(qa_doc_chain.run(input_document=chat, question="상황을 요약해줘"))