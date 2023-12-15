from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Initialising the ChatGPT LLM
chat = ChatOpenAI(temperature=0.0)

# Product Review - Input Data

review = """ Writing this review after using it for a couple of months now. It can take some time to get used to since the water jet is quite powerful. It might take you a couple of tries to get comfortable with some modes. Start with the teeth, get comfortable and then move on to the gums. Some folks may experience sensitivity. I experienced it for a day or so and then went away.
It effectively knocks off debris from between the teeth especially the hard to get like the fibrous ones. I haven't seen much difference in the tartar though. Hopefully, with time, it gets rid of it too.
There are 3 modes of usage: normal, soft and pulse. I started with soft then graduated to pulse and now use normal mode. For the ones who are not sure, soft mode is safe as it doesn't hit hard. Once you get used to the technique of holding and using the product, you could start experimenting with the other modes and choose the one that best suits you.
One time usage of the water full of tank should usually be sufficient if your teeth are relatively clean. If, however, you have hard to reach spaces with buildup etc. it might require a refill for a usage.
If you don't refill at all, one time full recharge of the battery in normal mode will last you 4 days with maximum strength of the water jet. If you refill it once, it'll last you 2 days after which the strength of the water jet reduces.
As for folks who are worried about the charging point getting wet, I accidentally used it once without the plug for the charging point and yet it worked fine and had no issues. Ideally keep the charging point covered with the plug provided with the product.
It has 2 jet heads (pink and blue) and hence the product can be used by 2 people as long as it's used hygienically. For charging, it comes with a USB cable without the adapter which shouldn't be an issue as your phone adapter should do the job.
I typically wash the product after every usage as the used water tends to run on the product during usage.
One issue I see is that the clasp for the water tank could break accidentally if not handled properly which will render the tank useless. So ensure to not keep it open unless you are filling the tank.
"""

# initialising the schema for each output
product_schema = ResponseSchema(name="Product",
                             description="Identify the product name")
review_schema = ResponseSchema(name="Review",
                                      description="Summarize the review")
sentiment_schema = ResponseSchema(name="Sentiment",
                                    description="Identify the sentiment of the review - positive/negative/neural")
topics_schema = ResponseSchema(name='Topics', description='Extract topics that user didnt like about the product.')

company_schema = ResponseSchema(name='Company',description='Identify the name of the company, if not then "not mentioned"')


# Concatinating the defined schema 
response_schemas = [product_schema,review_schema,sentiment_schema,topics_schema, company_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Generating the format instructions - that will be added to prompt
format_instructions = output_parser.get_format_instructions()

# Newly defined prompt that includes format instructions 

prompt = """
Your task is to provide insights for the product review \
on a e-commerce website, which is delimited by \
triple colons.

Perform following tasks:
1. Identify the product.
2. Summarize the product review, in upto 50 words.
3. Analyze the sentiment of review - positive/negative/neutral
4. Extract topics that user didnt like about the product.
5. Identify the name of the company, if not then "not mentioned"

Format using JSON keys

Product review: {text}

{format_instructions}
"""

prompt_template = ChatPromptTemplate.from_template(prompt)
review_message = prompt_template.format_messages(text = review, format_instructions= format_instructions)
output = chat(review_message)

# Passing the output through the parser

output_dict = output_parser.parse(output.content)

print(output_dict['Product'])