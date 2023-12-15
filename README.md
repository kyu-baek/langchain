안녕하세요. 
랭체인에 관심히 많은 백엔드 개발자 입니다.
오늘 제가 최종적으로 소개해드릴 서비스는 랭체인의 활용한 mbti 측정기 입니다.

해당 소스 코드는 Streamlit.py 입니다.

배포된 Streamlit 사이트 도메인입니다.

https://langchain-romina.streamlit.app/


처음 랭체인을 접하면서 새로운 개념에 적응하느라 고군분투했습니다. 

간단하게 openAi 에 질문을 보내는 것부터 시작해, 여러가지 랭체인 기능을 활용해보았습니다.
제가 랭체인을 공부한 여정을 정리해서 공유해보도록 하겠습니다.
코드는 파이썬으로 짰지만, 저는 파이썬을 그렇게 잘 다루지는 못합니다. 
그래서 코드가 조악한 점은 양해부탁드립니다.


단계별로 

OpenAi 를 사용하는 예시

ChatOpenAi 를 사용하는 예시

openAi 사용 요금 조회

연결된 질문을 기억하는 대화 체인을 사용하는 예시

Summary 체인을 사용해 글을 요약하는 예시

output parser 를 사용하는 예시

멀티 체인을 사용해 mbti 측정기 서비스 로직 구성 예시



OpenAi 를 사용하는 예시

from langchain.llms import OpenAI

llm = OpenAI(temperature=0.9)
text = "what would be a good company name for a company that makes colorful socks?"
print(llm(text))


ChatOpenAi 를 사용하는 예시

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(temperature=0.8)

# 여러개로 제너레이팅 가능
more_messages = [
[
    SystemMessage(content="You are a helpful assistant that translate English to Korean"),
    HumanMessage(content="Translate this sentence from English to Korean. I love programming.")
],
[
    SystemMessage(content="You are a helpful assistant that translate English to Korean"),
    HumanMessage(content="Translate this sentence from English to Korean. I hate programming.")
],
[
    SystemMessage(content="You are a helpful assistant that translate English to Korean"),
    HumanMessage(content="Translate this sentence from English to Korean. I do programming.")
]
]

result = chat.generate(more_messages)


**output**

generations=[[ChatGeneration(text='저는 프로그래밍을 사랑합니다.', generation_info={'finish_reason': 'stop'}, message=AIMessage(content='저는 프로그래밍을 사랑합니다.', additional_kwargs={}, example=False))], [ChatGeneration(text='나는 프로그래밍을 싫어해요.', generation_info={'finish_reason': 'stop'}, message=AIMessage(content='나는 프로그래밍을 싫어해요.', additional_kwargs={}, example=False))], [ChatGeneration(text='저는 프로그래밍을 합니다.', generation_info={'finish_reason': 'stop'}, message=AIMessage(content='저는 프로그래밍을 합니다.', additional_kwargs={}, example=False))]] llm_output={'token_usage': {'prompt_tokens': 99, 'completion_tokens': 40, 'total_tokens': 139}, 'model_name': 'gpt-3.5-turbo'} run=[RunInfo(run_id=UUID('4d03f852-e940-4c22-935a-ed58ff8fa956')), RunInfo(run_id=UUID('e3703ca4-92df-4da6-90fa-dd835f8e8c5e')), RunInfo(run_id=UUID('51d9e894-44a2-4704-b67a-d38b953efab4'))]
print(result)



3. openAi 사용 요금 조회


from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

chat = OpenAI(model_name='gpt-3.5-turbo')


# chat("1980년대 메탈 음악 5곡 추천해줘.")
with get_openai_callback() as cb:

  result = chat("1980년대 메탈 음악 5곡 추천해줘.")

  print(f"Total Tokens: {cb.total_tokens}")
  print(f"Prompt Tokens: {cb.prompt_tokens}")
  print(f"Completion Tokens: {cb.completion_tokens}")
  print(f"Total Cost (USD): ${cb.total_cost}")
  print(cb)

result



연결된 질문을 기억하는 대화 체인을 사용하는 예시

from langchain import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

llm = OpenAI(temperature=0)

conversation = ConversationChain(
    llm=llm, verbose=True, memory=ConversationBufferMemory()
)

conversation.predict(input="인공지능에서 Transformer 가 뭐야?")
print(conversation.memory)
print("\n")

conversation.predict(input="RNN과 차이를 설명해줘")
print(conversation.memory)

Summary 체인을 사용해 글을 요약하는 예시

from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

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

output parser 를 사용하는 예시

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser

class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""


    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

template = """You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more."""
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])
chain = chat_prompt | ChatOpenAI() | CommaSeparatedListOutputParser()

# 멀티스레드 환경에서 데이터 보호를 위해 Invoke 함수를 사용하는 것 같습니다.
chain.invoke({"text": "colors"})

**output**

['red', 'blue', 'green', 'yellow', 'orange']



7. 멀티 체인을 사용해 mbti 측정기 서비스 로직 구성 예시 with Streamlit

from langchain.chains import SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
import streamlit as st
import os



api_key = st.sidebar.text_input('OpenAI API Key', type='password')
os.environ['OPENAI_API_KEY'] = api_key
if api_key:

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
        "Chose one of the Overall MBTI types that repersent writer"
        "\n\n{summary}"
    )
    chain_three = LLMChain(llm=llm, prompt=third_prompt, 
                        output_key="mbti"
                        )



    # 1,2,3번 체인을 묶음
    overall_chain = SequentialChain(
        chains=[chain_one, chain_two, chain_three],
        input_variables=["Content"],
        output_variables=["summary", "sentiment", "mbti"],
    )



from langchain.callbacks import get_openai_callback
import re

st.title("MBTI 측정기")
Content = st.text_area("아직 친하진 않지만 당신이 궁금해하는 사람에 대해서 알려드릴게요. 그 사람이 작성한 sns 글을 올려주세요", height=200, max_chars=700)


if st.button('알아가기'):
    with get_openai_callback() as cb:
        result = overall_chain(Content, return_only_outputs=True)

    summary = result['summary']
    sentiment = result['sentiment']
    mbti = result['mbti']

    # 정규 표현식을 이용하여 처음으로 나오는 MBTI 유형을 찾음
    mbti_types = ["ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP", 
                "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"]

    pattern = r"\b(" + "|".join(mbti_types) + r")\b"
    mbti_result = re.search(pattern, mbti, re.IGNORECASE)

    st.subheader('summary')
    st.write(summary)
    st.divider()
    st.subheader('Predict :blue[MBTI]:')

    # 찾은 MBTI 결과를 저장
    if mbti_result:
        extracted_mbti = mbti_result.group(1).upper()  # 대문자로 변환
        st.header(extracted_mbti)
        st.write(mbti)
    else:
        st.write(mbti)
    st.divider()
    st.subheader('Predict :blue[Feeling now]:')
    st.title(sentiment)
    st.divider()
    st.info(cb)




MBTI 측정기를 만들어 보았으니, 얼마나 잘 측정해주는지 테스트를 해야겠죠?
아래는 일론 머스크의 인터뷰 글입니다. 

https://www.ft.com/content/697d8d32-6ef9-4b4c-835a-3e9dcbdb431a


일정 부분 발췌해서 MBTI 테스트에 넣어보겠습니다.



결과가 나왔습니다!!


일론 머스크의 인터뷰를 바탕으로 측정한 결과 그의 MBTI 는 INTJ 입니다.



다행이 실제 MBTI 와 일치하네요 😄


