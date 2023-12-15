from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain

llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key="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", temperature=0)

# stuff: LLM 한번에 다보냄. 길면 오류
# map_reduce : 나눠서 요약 후 합쳐서 다시 요약
# refine :(요약 + 다음문서) =>요약
# map_rerank: 점수를 매겨서 중요한거로 요약
summary_chain = load_summarize_chain(llm, chain_type="map_reduce")

from langchain.chains import AnalyzeDocumentChain

summary_doc_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)

with open('chat.txt') as f:
    chat = f.read()

print(summary_doc_chain.run(chat))