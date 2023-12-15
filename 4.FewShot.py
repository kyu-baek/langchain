from langchain import PromptTemplate, FewShotPromptTemplate


# few shot example
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
]

# next we specify the template to format the examples we have provided.
# we use the PromptTemplate class for this
example_formatter_template = """
Word: {word}
Antonym: {antonym}\n
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
    prefix="Give the antonym of word",

    # suffix 는 example 이후에 보내는 텍스트, 보통 유저 인풋이 어디로 가는지 지정한다.
    suffix="Word: {word}\nAntonym:",

    # 인풋 변수는 전체적인 프롬프트 기대값
    input_variables=["word"],

    # prefix, examples, suffix 에 join 되는 스트링
    example_separator="\n**\n",
)

print(few_shot_prompt.format(word="big"))