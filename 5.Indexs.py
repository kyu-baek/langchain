from langchain.document_loaders import TextLoader
loader = TextLoader('state_of_the_union.txt')

# 벡터 스토어에 텍스트로더 내용을 넣는다
from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator().from_loaders([loader])

query = "Whzt did the president say aout Ketanji Brown Jackson"
print(index.query(query))

# 쿼리와 답변 소스 정보를 줌. 어떤 문서를 사용했는지 참조를 위해서 씀
print(index.query_with_sources(query))
