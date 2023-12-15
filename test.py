from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


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
        SystemMessage(content="You are a psychological counselor who solves the problems between 갑 and 을. Please respond in the form of answer_form."),
        HumanMessage(content="""[{"id": 1, "writer": "갑", "content": "프사 왜 다 없었어? 상태명은 또 뭐야..우리 아직 헤어진 거 아니잖아.."}, {"id": 2, "writer": "을", "content": "없애든 말든 내 마음이야 너가 뭔 상관?"}, {"id": 3, "writer": "갑", "content": "그렇게 꼭 싸운 티를 내야겠어?"}, {"id": 4, "writer": "을", "content": "왜 이래라 저래라야? 싸운 거 티 좀 내면 어때? 내 공간인데 내 맘대로 사용하지도 못해?"}, {"id": 5, "writer": "갑", "content": "너 카톡에 우리 부모님도 다 있으시고 친구들도 다 있으니까 그렇지.. 프사랑 상태명 보고 매번 다들 연락와..헤어진거냐고"}, {"id": 6, "writer": "을", "content": "내 알바 아니야"}, {"id": 7, "writer": "갑", "content": "너 매번 싸울 때 마다 이러잖아..사람들한테 우리가 싸웠는지 헤어졌는지 다 광고 하고 싶어?"}]"""),
        SystemMessage(content="""[
  {
    "name": "answer_form",
    "description": "Understand the context of the conversation and kindly answer each question.",
    "parameters": {
      "type": "object",
      "properties": {
        "question_A": {
          "type": "array",
          "Lenght" : "around 2000 Korean words",
          "description": "In the conversation, measure one by each the overall emotional state of 갑 and 을 step by step.",
          "items": {
            "Target" : {
                "enum": ["갑", "을"]
            },
            "Measurement": "string",
        }
        "required": [
              "Target",
              "Measurement"
            ]
        },
        "question_B": {
            "type": "array",
            "Lenght" : "around 5000 Korean words",
            "description": "대화에서 내가 뭘 잘못했는지 잘 모르겠어. Step By Step 으로 상세하게 원인을 분석하고 해결책을 구체적으로 길게 제공해줘.",
            "items": {
                "Analysis": {
                "type": "string",
                "description": "분석 결과.",
            },
            "Solution": {
              "type": "string",
              "description": "해결책.",
            }
          },
          "required": [
            "Analysis",
            "Solution"
          ]
        }
      },
      "required": [
        "question_A",
        "question_B"
      ]
    }
  }
]"""),
    ]
]


chat = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="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", temperature=0)

# from langchain.chains import AnalyzeDocumentChain
# summary_chain = load_summarize_chain(chat, chain_type="map_reduce")
# summary_doc_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)


with get_openai_callback() as cb:


  result = chat.generate(more_messages)
#   summery = summary_chain.run(result)

  print(f"\n\nTotal Tokens: {cb.total_tokens}")
  print(f"Prompt Tokens: {cb.prompt_tokens}")
  print(f"Completion Tokens: {cb.completion_tokens}")
  print(f"Total Cost (USD): ${cb.total_cost}")
  print(cb)
  print("\n\n")

# result

# 아웃풋
#  generations=[[ChatGeneration(text='저는 프로그래밍을 사랑합니다.', generation_info={'finish_reason': 'stop'}, message=AIMessage(content='저는 프로그래밍을 사랑합니다.', additional_kwargs={}, example=False))], [ChatGeneration(text='나는 프로그래밍을 싫어해요.', generation_info={'finish_reason': 'stop'}, message=AIMessage(content='나는 프로그래밍을 싫어해요.', additional_kwargs={}, example=False))], [ChatGeneration(text='저는 프로그래밍을 합니다.', generation_info={'finish_reason': 'stop'}, message=AIMessage(content='저는 프로그래밍을 합니다.', additional_kwargs={}, example=False))]] llm_output={'token_usage': {'prompt_tokens': 99, 'completion_tokens': 40, 'total_tokens': 139}, 'model_name': 'gpt-3.5-turbo'} run=[RunInfo(run_id=UUID('4d03f852-e940-4c22-935a-ed58ff8fa956')), RunInfo(run_id=UUID('e3703ca4-92df-4da6-90fa-dd835f8e8c5e')), RunInfo(run_id=UUID('51d9e894-44a2-4704-b67a-d38b953efab4'))]
print(result)
print("\n\n")
# print(summery)
