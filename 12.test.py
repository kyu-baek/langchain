import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import warnings
warnings.filterwarnings("ignore")

llm_model = "gpt-3.5-turbo"

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from langchain.chains import SequentialChain

llm = ChatOpenAI(temperature=0.9, openai_api_key="sk-", model=llm_model)

first_prompt = ChatPromptTemplate.from_template(
    """
    ```function
    {function}
    ```
    The function you entered is a C language function. Please describe the function.
    """
)

chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="description")

second_prompt = ChatPromptTemplate.from_template(
    """
    ```function
    {function}
    ```
    The function you entered is a C language function. Omit the description of the function and explain how it is used in computer science. For example, if it is printf, explain how it works internally when multiple printf functions are written to the standard output.
    """
)

chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="usage")

third_prompt = ChatPromptTemplate.from_template(
    """
    ```function
    {function}
    ```
    ```description
    {description}
    ```
    ```usage
    {usage}
    ```
    Please write an example code based on function,description,usage.
    """
)

chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key="example")

fourth_prompt = ChatPromptTemplate.from_template(
    """
    ```function
    {function}
    ```
    ```description
    {description}
    ```
    ```usage
    {usage}
    ```
    ```example
    {example}
    ```
    Please translate to Korean. Please do not translate jargon.
    """
)

llm_model_16k = "gpt-3.5-turbo-16k-0613"
llm_16k = ChatOpenAI(temperature=0.5, model=llm_model_16k)

chain_four = LLMChain(llm=llm_16k, prompt=fourth_prompt, output_key="translate")

five_prompt = ChatPromptTemplate.from_template(
    """
    {translate}
    Organize it in markdown.
    """
)

chain_five = LLMChain(llm=llm, prompt=five_prompt, output_key="markdown")

overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four, chain_five],
    input_variables=["function"],
    output_variables=["description", "usage", "example", "translate", "markdown"],
    verbose=True
)

function = "open"
data = {'function': function}
result = overall_chain(data)
print(result['markdown'])

def save_to_file(text, filename):
    with open(filename, 'w') as file:
        file.write(text)

save_to_file(result['markdown'], 'example_output.md')
