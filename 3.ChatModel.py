from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(temperature=0.8)

# chat([HumanMessage(content="Translate this sentence from English to Korean. I love programming.")])
# 아웃풋 -> content='나는 프로그래밍을 사랑합니다.' additional_kwargs={} example=False

# messages = [
#     SystemMessage(content="You are a helpful assistant that translate English to Korean"),
#     HumanMessage(content="Translate this sentence from English to Korean. I love programming.")
# ]
# print(chat(messages))
# 아웃풋 -> content='나는 프로그래밍을 사랑합니다.' additional_kwargs={} example=False


# 여러개로 제너레이팅 하고 싶다하면 
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


# 아웃풋
#  generations=[[ChatGeneration(text='저는 프로그래밍을 사랑합니다.', generation_info={'finish_reason': 'stop'}, message=AIMessage(content='저는 프로그래밍을 사랑합니다.', additional_kwargs={}, example=False))], [ChatGeneration(text='나는 프로그래밍을 싫어해요.', generation_info={'finish_reason': 'stop'}, message=AIMessage(content='나는 프로그래밍을 싫어해요.', additional_kwargs={}, example=False))], [ChatGeneration(text='저는 프로그래밍을 합니다.', generation_info={'finish_reason': 'stop'}, message=AIMessage(content='저는 프로그래밍을 합니다.', additional_kwargs={}, example=False))]] llm_output={'token_usage': {'prompt_tokens': 99, 'completion_tokens': 40, 'total_tokens': 139}, 'model_name': 'gpt-3.5-turbo'} run=[RunInfo(run_id=UUID('4d03f852-e940-4c22-935a-ed58ff8fa956')), RunInfo(run_id=UUID('e3703ca4-92df-4da6-90fa-dd835f8e8c5e')), RunInfo(run_id=UUID('51d9e894-44a2-4704-b67a-d38b953efab4'))]
print(result)
