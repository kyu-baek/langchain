from langchain.llms import OpenAI


llm = OpenAI(temperature=0.9)
text = "what would be a good company name for a company that makes colorful socks?"
print(llm(text))

