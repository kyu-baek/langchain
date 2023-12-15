import openai

# open ai에서 발급받은 api key를 등록합니다
OPENAI_YOUR_KEY = "sk-"
openai.api_key = OPENAI_YOUR_KEY

MODEL = "gpt-3.5-turbo"
USER_INPUT_MSG = "[{\"id\": 1, \"writer\": \"갑\", \"content\": \"프사 왜 다 없었어? 상태명은 또 뭐야..우리 아직 헤어진 거 아니잖아..\"}, {\"id\": 2, \"writer\": \"을\", \"content\": \"없애든 말든 내 마음이야 너가 뭔 상관?\"}, {\"id\": 3, \"writer\": \"갑\", \"content\": \"그렇게 꼭 싸운 티를 내야겠어?\"}, {\"id\": 4, \"writer\": \"을\", \"content\": \"왜 이래라 저래라야? 싸운 거 티 좀 내면 어때? 내 공간인데 내 맘대로 사용하지도 못해?\"}, {\"id\": 5, \"writer\": \"갑\", \"content\": \"너 카톡에 우리 부모님도 다 있으시고 친구들도 다 있으니까 그렇지.. 프사랑 상태명 보고 매번 다들 연락와..헤어진거냐고\"}, {\"id\": 6, \"writer\": \"을\", \"content\": \"내 알바 아니야\"}, {\"id\": 7, \"writer\": \"갑\", \"content\": \"너 매번 싸울 때 마다 이러잖아..사람들한테 우리가 싸웠는지 헤어졌는지 다 광고 하고 싶어?\"}]"
# ANSWER_FORM = "[{\"name\": \"answer_form\", \"description\": \"대화 내용의 맥락을 이해하고 '을'의 입장을 공감한 상태에서 각각의 질문에 대해 친절하게 답변해줘.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"question_A\": {\"type\": \"array\", \"array_size\": 543, \"description\": \"각 id에 해당하는 작성자의 감정 상태를 low, medium, high 중에서 선택해줘.\", \"items\": {\"type\": \"string\", \"enum\": [\"low\", \"medium\", \"high\"]}}, \"question_B\": {\"type\": \"array\", \"description\": \"각각의 id의 대화 내용에서 숨은 의도를 분석해줘, 그리고 대화의 쟁점을 찾아서 id별로 분석한 결과와 해결책을 친절하게 제시해줘.\", \"array_size\": 543, \"items\": {\"id\": {\"type\": \"integer\", \"description\": \"대화별 id\"}, \"analysis\": {\"type\": \"string\", \"description\": \"분석 결과.\", \"limit_length\": 100}, \"solution\": {\"type\": \"string\", \"description\": \"해결책.\", \"limit_length\": 100}, \"question_C\": {\"type\": \"array\", \"description\": \"각각의 id의 대화 내용에서 유치한 말투가 있는지 궁금해. 이를 분석해줘.\", \"array_size\": 543, \"items\": {\"id\": {\"type\": \"integer\", \"description\": \"대화별 id\"}, \"analysis\": {\"type\": \"string\", \"description\": \"분석 결과.\", \"limit_length\": 100}, \"required\": [\"id\", \"analysis\"]}}, \"required\": [\"question_A\", \"question_B\", \"question_C\"]}}]"
ANSWER_FORM = """
[
  {
    "name": "answer_form",
    "description": "Understand the context of the conversation and kindly answer each question.",
    "parameters": {
      "type": "object",
      "properties": {
        "question_A": {
          "type": "array",
          "Length" : around 200 Korean words,
          "description": "In the conversation, measure the overall emotional state of 갑 and 을 one by each.",
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
            "Length" : around 800 Korean words,
            "description": "대화에서 말투의 문제점을 찾아서 Step By Step 으로 상세하게 원인을 분석하고 해결책을 구체적으로 길게 제공해줘.",
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
]"""
print("Answer Form = \n")
print(ANSWER_FORM)
response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Respond in the form of answer_form."},
        {"role": "user", "content": USER_INPUT_MSG}, 
        {"role": "system", "content": ANSWER_FORM},
    ],
    temperature=0.7,
)
print("Answer Result = \n")
print(response['choices'][0]['message']['content'])

        # {"role": "system", "content": "You are a psychological counselor who solves the problems between 갑 and 을. Please respond in the form of answer_form."},
