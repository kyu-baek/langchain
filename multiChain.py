from langchain.chains import SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
import os

os.environ['OPENAI_API_KEY'] = 'sk-'




llm = ChatOpenAI(temperature=0.8, model="gpt-3.5-turbo-0613")


# 1. input 값 요약하기
first_prompt = ChatPromptTemplate.from_template(
    "Summarize the content:"
    "\n\n{Content}"
)
chain_one = LLMChain(llm=llm, prompt=first_prompt, 
                     output_key="summary"
                    )


# 2. 감정상태 이모지로 표현하기
second_prompt = ChatPromptTemplate.from_template(
    "Chose one of the emogis in the Emogi that repersent writer's sentiment"
    "\n\n{summary}"
)
chain_two = LLMChain(llm=llm, prompt=second_prompt, 
                     output_key="sentiment"
                    )

# 3. MBTI 추측하기
third_prompt = ChatPromptTemplate.from_template(
    "Chose one of the MBTI types that repersent writer and Tell Overall"
    "\n\n{summary}"
)
chain_three = LLMChain(llm=llm, prompt=third_prompt, 
                     output_key="mbti"
                    )



# overall_chain: input= Content 
# and output= summary,sentiment, mbti
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three],
    input_variables=["Content"],
    output_variables=["summary", "sentiment", "mbti"],
    verbose=False,
)

Content = """
이직 한지 한달 좀 넘었는데
회사 고민 계속 되서 조언 부탁해!

화장품 회사인줄 알았는데
pb만 파는게 아니라 돈되는 건 다 파는 회사.
처우나 조건은 나쁘지 않으나
규모가 작아 체계가 없고
체계가 잡히려면 시간이 꽤나 들것으로 보임 +
7시 퇴근인데 평일 저녁 시간 활용이 어려운게 아쉬움+
업무는 외부몰 md인데 현재 cs, 발주, 송장 업무도 같이하고 있음 > 처음엔 힘들어 죽겠었는데 한달 지나니
이것도 적응해서 어떻게 해가고는 있음
근데 주말에 쉬는데 내내 마음이 편하지가 않아서
그만 둬야되는건가 계속 고민중
(일은 힘들어도 마음은 편해야되는게 아닌가 생각)
"""

from langchain.callbacks import get_openai_callback
with get_openai_callback() as cb:
    result = overall_chain(Content, return_only_outputs=True)
    print(cb)
    print(result)

