#@title 1. API 사용 금액 조회

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