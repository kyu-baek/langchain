from langchain import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

llm = OpenAI(temperature=0)

conversation = ConversationChain(
    llm=llm, verbose=True, memory=ConversationBufferMemory()
)

# conversation = ConversationChain(llm=llm, verbose=True)
conversation.predict(input="인공지능에서 Transformer 가 뭐야?")
print(conversation.memory)
print("\n")

conversation.predict(input="RNN과 차이를 설명해줘")
print(conversation.memory)

