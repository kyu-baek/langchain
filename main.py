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
Content = st.text_area("아직 친하진 않지만 당신이 궁금해하는 사람에 대해서 알려드릴게요. 그 사람이 작성한 sns 글을 올려주세요", height=200, max_chars=2000)


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
