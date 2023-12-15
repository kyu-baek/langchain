from langchain import PromptTemplate, FewShotPromptTemplate


# few shot example
examples = [
    {"question": "각각의 id의 대화 내용에서 문제점을 파악하고, 그 문제를 해결할 수 있는 방안을 id별로 분석한 결과와 해결책을 친절하게 제시해줘", "answer": """{
    "answer_form": {
        "question_A": [
            "medium",
            "low",
            "medium",
            "high",
            "high",
            "low",
            "high"
        ],
        "question_B": [
            {
                "solution": "갑이가 을의 카카오톡 프사와 상태명에 대해 강요하지 않고, 서로의 공간과 자유를 존중하는 마음가짐으로 대화를 이어나가면 좋을 것 같아요.",
                "id": 1,
                "analysis": "갑이가 을에게 프사와 상태명을 바꾸라고 요구하는 것은 조금 과한 요구인 것 같아요. 을은 자신의 마음을 표현하기 위해 자신의 카카오톡을 사용하는 거니까요."
            },
            {
                "solution": "갑이는 을의 개인적인 욕구에 대해 이해하고 존중해줄 필요가 있어요. 어떤 의사소통 방법이 서로에게 가장 편한지 이야기하고 합의점을 찾아보는 것이 좋을 것 같아요.",
                "id": 2,
                "analysis": "을이가 갑에게 프사와 상태명에 신경을 쓰거나 싸움을 통해 공개적으로 알림을 받고 싶어하는 이유가 있을 수 있겠네요. 이것은 을이의 개인적인 욕구일 수 있습니다."
            },
            {
                "solution": "가족이나 친구, 그리고 사회적 관계에서 갑이와 을이가 싸운 사실을 알리지 않는 것이 좋을 것 같아요. 이렇게 함으로써 서로의 관계를 지키고 좋은 방향으로 나아갈 수 있을 거에요.",
                "id": 3,
                "analysis": "갑이와 을이는 싸울 때마다 싸운 사실을 다른 사람들에게 알리려는 경향이 있습니다. 이는 자존감 문제와 같은 복잡한 감정을 표출하려는 행동일 수 있습니다."
            },
            {
                "solution": "갑이는 을이의 개인 공간에 대해 좀 더 이해하고 존중할 필요가 있어요. 서로의 알바에 대한 예의와 존중을 지켜가며 상호간의 공간을 확보하는 것이 중요해요.",
                "id": 4,
                "analysis": "을이는 자신의 개인 공간에 대해 좀 더 강하게 주장하는 것 같아요. 을이에게는 자신의 알바에 대한 소유권이 있으며, 그 공간을 자유롭게 사용하고 싶어하는 욕구가 있을 것 같습니다."
            },
            {
                "solution": "갑이는 을이의 걱정을 이해하고, 친구들에게 싸웠거나 헤어진 것은 아니라고 알릴 수 있는 방법을 찾아보는 것이 좋을 것 같아요. 서로의 관계를 존중하고 이해할 수 있는 방식으로 대화를 이어갈 필요가 있어요.",
                "id": 5,
                "analysis": "갑이는 을이의 카카오톡에 우리 부모님과 친구들이 있어서 을이의 프사와 상태명을 신중하게 선택하라고 말한 것 같아요. 이는 을이의 친구들과의 관계와 을이의 이미지에 대한 걱정일 수 있습니다."
            },
            {
                "solution": "갑이는 을이의 익명성을 존중하고 이해해줘야 해요. 민감한 주제에 대해 익명의 대화 상대가 안전하고 편안한 상황에서 대화할 수 있도록 배려하는 것이 중요해요.",
                "id": 6,
                "analysis": "익명의 대화 상대와의 대화에서는 자신을 밝히지 않는 것이 일반적일 수 있습니다. 을이는 자신의 신원을 밝히지 않는 것으로 갑이와의 대화를 보호하려는 것이 아닌가 싶어요."
            },
            {
                "solution": "갑이와 을이는 서로의 프라이버시와 존엄을 존중하고 자상하게 대화해야 해요. 싸움을 알리는 행동 대신에 서로를 이해하고 용서하는 마음을 갖는 것이 좋을 것 같아요.",
                "id": 7,
                "analysis": "갑이와 을이는 자주 싸울 때마다 싸운 사실을 다른 사람들에게 알리는 경향이 있습니다. 이번 상황은 서로가 상대방을 괴롭히고자 하는 의도가 있을 수 있습니다."
            }
        ]
    }
}"""},
    {},
]

# next we specify the template to format the examples we have provided.
# we use the PromptTemplate class for this
example_formatter_template = """
Chat: {word}
Answer: {antonym}\n
"""

example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_formatter_template,
)

# Finally we create the FewShotPromptTemplate object.
few_shot_prompt = FewShotPromptTemplate(
    # 우리가 넣고 싶은 인풋 예시
    examples=examples,

    # 우리가 인풋을 넣을 때 지정하고 싶은 포맷
    example_prompt=example_prompt,

    # prefix 는 우리가 example 을 넣기 전에 보낼 텍스트, 보통 consists of intructions 함
    prefix="Give the antonym of every input",

    # suffix 는 example 이후에 보내는 텍스트, 보통 유저 인풋이 어디로 가는지 지정한다.
    suffix="Word: {input}\nAntonym:",

    # 인풋 변수는 전체적인 프롬프트 기대값
    input_variables=["input"],

    # prefix, examples, suffix 에 join 되는 스트링
    example_separator="\n**\n",
)

print(few_shot_prompt.format(input="big"))