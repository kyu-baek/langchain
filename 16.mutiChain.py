from langchain.chains import SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(temperature=0.0)

# prompt template 1: summarize the product review - output (summary)
first_prompt = ChatPromptTemplate.from_template(
    "Summarize the product review:"
    "\n\n{Review}"
)

# chain 1: input= Review and output= summary
chain_one = LLMChain(llm=llm, prompt=first_prompt, 
                     output_key="summary"
                    )

# prompt template 2: identify the sentiment from the summary
second_prompt = ChatPromptTemplate.from_template(
    "Identify the sentiment of the review"
    "\n\n{summary}"
)
# chain 2: input= summary and output= sentiment
chain_two = LLMChain(llm=llm, prompt=second_prompt, 
                     output_key="sentiment"
                    )

# prompt template 3: identify the topics that are causing issues
third_prompt = ChatPromptTemplate.from_template(
    "Identify the topics which are causing issues for customer:\n\n{Review}"
)
# chain 3: input= Review and output= topics
chain_three = LLMChain(llm=llm, prompt=third_prompt,
                       output_key="topics"
                      )


# prompt template 4: follow up message
fourth_prompt = ChatPromptTemplate.from_template(
    """Write a follow up response to the customer based on sentiment
      and topics:
    "\n\nSentiment: {sentiment}\n\Topics: {topics}"""
)
# chain 4: input= sentiment, topics and output= followup_message
chain_four = LLMChain(llm=llm, prompt=fourth_prompt,
                      output_key="followup_message"
                     )

# overall_chain: input= Review 
# and output= summary,sentiment, topics, followup_message
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["summary", "sentiment","topics","followup_message"],
    verbose=False
)

review = """
I never thought that I would receive such a bad iPhone from Amazon when I ordered it online. My iPhone is taking extremely blurry photos and it heats up quickly, within 10 minutes of using Instagram or snap. when I called customer service for a replacement,they refused to exchange it directly. This is the first time I bought an iPhone and I had a hard time saving up for it. I thought the camera would be great, but it is taking very blurry photos and overheating. I bought it for Rs. 60,499 because there was a bigger discount than on Flinkart but now I regret buying it from here because I think the seller sold it to me for a lower price due to their own faulty iPhone 13." My only request is that you replace my iPhone 13 as soon as possible
"""
overall_chain(review)