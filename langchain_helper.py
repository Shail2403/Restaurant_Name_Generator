import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from secret_key import openapi_key  # Importing your API key from secret_key.py

# Set the Google API key from the imported variable
os.environ['GOOGLE_API_KEY'] = openapi_key

# Initialize the LLM with the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-pro")


def generate_restaurant_name_and_items(cuisine):
    # Create the prompt templates
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open a restaurant for {cuisine} food. Suggest a one fancy name for this."
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template="Suggest some menu items for {restaurant_name}. Return it as a comma-separated list.Seperately after this At the end  randomly tell Most Popular Item  anyone from menu in good looking ui ux and inside box colourful sttractive way,need only name nothing else  show that name in good looking dark box and large size font.No need needless description or text."
    )

    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")

    # Create the sequential chain
    chain = SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', 'menu_items']
    )

    # Run the chain
    response = chain({'cuisine': cuisine})

    return response


if __name__ == '__main__':
    print(generate_restaurant_name_and_items("Italian"))
