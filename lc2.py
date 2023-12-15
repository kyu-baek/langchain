from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["product"],
    template="what would be a good company name for a company that makes {product}?",
)

print(prompt.format(product="colorful"))