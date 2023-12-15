from langchain.llms import OpenAI


llm = OpenAI(temperature=0.9)

llm_result = llm.generate(["Tell me a joke", "Tell me a poem"]*15)


# 토큰 사용 정보 출력 가능
# {
#     'token_usage': {
#        'prompt_tokens': 120,
#        'total_tokens': 982, 
#        'completion_tokens': 862
#      }, 
#     'model_name': 'text-davinci-003'
# }
print(llm_result.llm_output)

# 개수
print(len(llm_result.generations))